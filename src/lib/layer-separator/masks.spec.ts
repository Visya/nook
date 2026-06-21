import { describe, it, expect } from 'vitest';
import {
	evenLayers,
	assignPixelsToLayers,
	buildCumulativeMasks,
	buildLayerMasks,
	buildLayerCutout,
	featherMask,
	expandMask,
	applyEdge,
	applyEdgeToMasks,
	depthToLayerMasks,
	depthToMasks,
	evenThresholds,
	layersFromThresholds,
	depthHistogram,
	resizeThresholds
} from './masks';

describe('evenLayers', () => {
	it('rejects fewer than 2 layers', () => {
		expect(() => evenLayers(1)).toThrow();
	});

	it('covers the full depth range', () => {
		const layers = evenLayers(3);
		expect(layers[0].depthMin).toBe(0);
		expect(layers[layers.length - 1].depthMax).toBe(256);
	});

	it('partitions the range without gaps or overlap', () => {
		const layers = evenLayers(4);
		for (let i = 1; i < layers.length; i++) {
			expect(layers[i].depthMin).toBe(layers[i - 1].depthMax);
		}
	});

	it('produces empty overrides', () => {
		const layers = evenLayers(3);
		for (const l of layers) expect(l.overrides).toEqual([]);
	});
});

describe('assignPixelsToLayers', () => {
	it('puts pixel in the correct depth bin', () => {
		const layers = evenLayers(3);
		// 3 layers → [0..85), [85..171), [171..256)
		const depth = new Uint8Array([0, 50, 85, 100, 170, 171, 200, 255]);
		const result = assignPixelsToLayers(depth, layers);
		expect(Array.from(result)).toEqual([0, 0, 1, 1, 1, 2, 2, 2]);
	});

	it('applies override masks over depth thresholds', () => {
		const layers = evenLayers(3);
		// Force pixel 0 (which depth says is layer 0) into layer 2.
		layers[2].overrides.push({
			source: 'sam-override',
			mask: new Uint8Array([255, 0, 0, 0])
		});
		const depth = new Uint8Array([0, 0, 200, 200]);
		const result = assignPixelsToLayers(depth, layers);
		expect(Array.from(result)).toEqual([2, 0, 2, 2]);
	});

	it('lets a later layer override an earlier override on the same pixel', () => {
		const layers = evenLayers(3);
		layers[1].overrides.push({ source: 'sam-override', mask: new Uint8Array([255]) });
		layers[2].overrides.push({ source: 'sam-override', mask: new Uint8Array([255]) });
		const depth = new Uint8Array([0]);
		expect(Array.from(assignPixelsToLayers(depth, layers))).toEqual([2]);
	});

	it('throws when an override mask length mismatches depth', () => {
		const layers = evenLayers(2);
		layers[0].overrides.push({ source: 'sam-override', mask: new Uint8Array([255, 255]) });
		expect(() => assignPixelsToLayers(new Uint8Array([0, 0, 0]), layers)).toThrow();
	});

	it('subtract override pushes its region from the layer to the layer behind', () => {
		const layers = evenLayers(3);
		// Depth puts pixels 2,3 in layer 2 (near). Subtract them from layer 2 → fall to layer 1.
		layers[2].overrides.push({
			source: 'sam-override',
			op: 'subtract',
			mask: new Uint8Array([0, 0, 255, 255])
		});
		const depth = new Uint8Array([0, 100, 200, 255]); // → layers [0, 1, 2, 2]
		expect(Array.from(assignPixelsToLayers(depth, layers))).toEqual([0, 1, 1, 1]);
	});

	it('subtract only affects pixels actually assigned to that layer', () => {
		const layers = evenLayers(3);
		// Mask covers pixels 0 (layer 0) and 2 (layer 2); subtracting from layer 2 leaves pixel 0 alone.
		layers[2].overrides.push({
			source: 'sam-override',
			op: 'subtract',
			mask: new Uint8Array([255, 0, 255, 0])
		});
		const depth = new Uint8Array([0, 100, 200, 255]); // → [0, 1, 2, 2]
		expect(Array.from(assignPixelsToLayers(depth, layers))).toEqual([0, 1, 1, 2]);
	});

	it('add then subtract: object forced into a layer can be carved back out', () => {
		const layers = evenLayers(3);
		layers[2].overrides.push({ source: 'sam-override', op: 'add', mask: new Uint8Array([255, 0]) });
		layers[2].overrides.push({
			source: 'sam-override',
			op: 'subtract',
			mask: new Uint8Array([255, 0])
		});
		const depth = new Uint8Array([0, 0]); // depth → layer 0
		// add puts pixel 0 in layer 2, subtract then pushes it back to layer 1.
		expect(Array.from(assignPixelsToLayers(depth, layers))).toEqual([1, 0]);
	});
});

describe('buildCumulativeMasks', () => {
	it('returns N-1 masks for N layers', () => {
		const pixelLayers = new Uint8Array([0, 1, 2]);
		expect(buildCumulativeMasks(pixelLayers, 3)).toHaveLength(2);
	});

	it('returns empty for fewer than 2 layers', () => {
		expect(buildCumulativeMasks(new Uint8Array([0]), 1)).toEqual([]);
	});

	it('mask k is black for pixels in layers 0..k, white for k+1..N-1', () => {
		// 4 pixels in layers [0, 1, 2, 2] across 3 layers → 2 masks
		const pixelLayers = new Uint8Array([0, 1, 2, 2]);
		const [m0, m1] = buildCumulativeMasks(pixelLayers, 3);
		// m0: layer 0 black, layers 1+2 white
		expect(Array.from(m0)).toEqual([0, 255, 255, 255]);
		// m1: layers 0+1 black, layer 2 white
		expect(Array.from(m1)).toEqual([0, 0, 255, 255]);
	});

	it('matches the contract on the README example (sky, mid, foreground)', () => {
		// Pretend 6 pixels: sky, sky, mid, mid, fg, fg → layers 0, 0, 1, 1, 2, 2
		const pixelLayers = new Uint8Array([0, 0, 1, 1, 2, 2]);
		const [maskSky, maskSkyPlusMid] = buildCumulativeMasks(pixelLayers, 3);
		// maskSky: sky black, the rest white
		expect(Array.from(maskSky)).toEqual([0, 0, 255, 255, 255, 255]);
		// maskSkyPlusMid: sky+mid black, foreground white
		expect(Array.from(maskSkyPlusMid)).toEqual([0, 0, 0, 0, 255, 255]);
	});
});

describe('depthToMasks (integration)', () => {
	it('chains assignment + cumulative correctly', () => {
		const layers = evenLayers(3);
		const depth = new Uint8Array([0, 100, 200]);
		const masks = depthToMasks(depth, layers);
		expect(masks).toHaveLength(2);
		// pixel layers: [0, 1, 2]
		expect(Array.from(masks[0])).toEqual([0, 255, 255]);
		expect(Array.from(masks[1])).toEqual([0, 0, 255]);
	});
});

describe('buildLayerMasks', () => {
	it('returns N isolated masks for N layers', () => {
		const pixelLayers = new Uint8Array([0, 1, 2]);
		expect(buildLayerMasks(pixelLayers, 3)).toHaveLength(3);
	});

	it('mask k is white only for pixels in layer k', () => {
		const pixelLayers = new Uint8Array([0, 1, 2, 2]);
		const [m0, m1, m2] = buildLayerMasks(pixelLayers, 3);
		expect(Array.from(m0)).toEqual([255, 0, 0, 0]);
		expect(Array.from(m1)).toEqual([0, 255, 0, 0]);
		expect(Array.from(m2)).toEqual([0, 0, 255, 255]);
	});

	it('masks are disjoint and complete (each pixel white in exactly one, OR == all-255)', () => {
		const pixelLayers = new Uint8Array([0, 0, 1, 2, 2, 1]);
		const masks = buildLayerMasks(pixelLayers, 3);
		for (let i = 0; i < pixelLayers.length; i++) {
			const white = masks.filter((m) => m[i] === 255).length;
			expect(white).toBe(1);
			expect(masks.reduce((acc, m) => acc | m[i], 0)).toBe(255);
		}
	});
});

describe('featherMask', () => {
	it('radius 0 returns an unchanged copy', () => {
		const mask = new Uint8Array([0, 255, 255, 0]);
		const out = featherMask(mask, 2, 2, 0);
		expect(Array.from(out)).toEqual([0, 255, 255, 0]);
		expect(out).not.toBe(mask);
	});

	it('keeps a fully-white interior at 255', () => {
		// 5x5 all white → every window is all-255 → stays 255.
		const mask = new Uint8Array(25).fill(255);
		const out = featherMask(mask, 5, 5, 1);
		expect(Array.from(out)).toEqual(Array(25).fill(255));
	});

	it('keeps a border-touching white region opaque at the frame', () => {
		const mask = new Uint8Array(25).fill(255);
		const out = featherMask(mask, 5, 5, 2);
		// Corner and edge pixels replicate the border, so they stay white.
		expect(out[0]).toBe(255);
		expect(out[4]).toBe(255);
		expect(out[24]).toBe(255);
	});

	it('ramps monotonically across a vertical half-plane boundary', () => {
		// Left 3 columns white, right 3 black, 6 wide × 1 tall.
		const width = 6;
		const mask = new Uint8Array([255, 255, 255, 0, 0, 0]);
		const out = featherMask(mask, width, 1, 1);
		for (let x = 1; x < width; x++) {
			expect(out[x]).toBeLessThanOrEqual(out[x - 1]);
		}
		// Far interior on each side stays saturated.
		expect(out[0]).toBe(255);
		expect(out[width - 1]).toBe(0);
	});
});

describe('expandMask', () => {
	it('radius 0 returns an unchanged copy', () => {
		const mask = new Uint8Array([0, 255, 0, 0]);
		const out = expandMask(mask, 2, 2, 0);
		expect(Array.from(out)).toEqual([0, 255, 0, 0]);
		expect(out).not.toBe(mask);
	});

	it('grows a single white pixel into its neighborhood with hard edges', () => {
		// 5x5, single white center at index 12. Dilate by 1 → 3x3 block white, rest black.
		const mask = new Uint8Array(25);
		mask[12] = 255;
		const out = expandMask(mask, 5, 5, 1);
		const white = new Set([6, 7, 8, 11, 12, 13, 16, 17, 18]);
		for (let i = 0; i < 25; i++) {
			expect(out[i]).toBe(white.has(i) ? 255 : 0);
		}
	});

	it('only produces 0 or 255 (hard edge, no soft ramp)', () => {
		const mask = new Uint8Array([255, 255, 0, 0, 0, 0]);
		const out = expandMask(mask, 6, 1, 1);
		for (const v of out) expect(v === 0 || v === 255).toBe(true);
		// The black side adjacent to white grows by one pixel.
		expect(out[2]).toBe(255);
		expect(out[3]).toBe(0);
	});
});

describe('applyEdge', () => {
	it('dispatches to feather (soft) vs expand (hard)', () => {
		const mask = new Uint8Array([255, 255, 255, 0, 0, 0]);
		const soft = applyEdge(mask, 6, 1, 1, 'feather');
		const hard = applyEdge(mask, 6, 1, 1, 'expand');
		expect(soft.some((v) => v > 0 && v < 255)).toBe(true); // ramp
		for (const v of hard) expect(v === 0 || v === 255).toBe(true); // crisp
	});

	it('radius 0 is identity (copy) for both modes', () => {
		const mask = new Uint8Array([0, 255, 0, 255]);
		expect(Array.from(applyEdge(mask, 2, 2, 0, 'feather'))).toEqual([0, 255, 0, 255]);
		expect(Array.from(applyEdge(mask, 2, 2, 0, 'expand'))).toEqual([0, 255, 0, 255]);
	});
});

describe('applyEdgeToMasks', () => {
	it('returns the same array reference when radius is 0', () => {
		const masks = [new Uint8Array([0, 255]), new Uint8Array([255, 0])];
		expect(applyEdgeToMasks(masks, 2, 1, 0, 'feather')).toBe(masks);
	});

	it('applies the edge to every mask', () => {
		const masks = [new Uint8Array([255, 0, 0, 0]), new Uint8Array([0, 0, 0, 255])];
		const out = applyEdgeToMasks(masks, 4, 1, 1, 'expand');
		expect(out[0][1]).toBe(255); // grew right from pixel 0
		expect(out[1][2]).toBe(255); // grew left from pixel 3
	});
});

describe('buildLayerCutout', () => {
	it('copies colour straight through and uses the mask as alpha', () => {
		// 2x1 image: red, green. Mask keeps pixel 0 fully, drops pixel 1.
		const rgba = new Uint8ClampedArray([255, 0, 0, 255, 0, 255, 0, 255]);
		const mask = new Uint8Array([255, 0]);
		const out = buildLayerCutout(rgba, mask, 2, 1);
		expect(out.length).toBe(8);
		expect(Array.from(out.slice(0, 4))).toEqual([255, 0, 0, 255]);
		// Colour preserved even where alpha is 0.
		expect(Array.from(out.slice(4))).toEqual([0, 255, 0, 0]);
	});

	it('carries a feathered (partial) mask value into alpha', () => {
		const rgba = new Uint8ClampedArray([10, 20, 30, 255]);
		const out = buildLayerCutout(rgba, new Uint8Array([128]), 1, 1);
		expect(out[3]).toBe(128);
	});

	it('invert flips the alpha — keeps the black side of a cumulative mask (background-preserving)', () => {
		// Cumulative mask: pixel 0 black (=this band + behind), pixel 1 white (=in front).
		const rgba = new Uint8ClampedArray([255, 0, 0, 255, 0, 255, 0, 255]);
		const mask = new Uint8Array([0, 255]);
		const out = buildLayerCutout(rgba, mask, 2, 1, true);
		expect(out[3]).toBe(255); // background pixel kept
		expect(out[7]).toBe(0); // foreground pixel dropped
	});

	it('stacking the N cutouts back-to-front reconstructs the source', () => {
		const rgba = new Uint8ClampedArray([10, 20, 30, 255, 40, 50, 60, 255, 70, 80, 90, 255]);
		const layers = layersFromThresholds([85, 170]); // 3 layers
		const depth = new Uint8Array([0, 100, 200]); // → layers [0, 1, 2]
		const masks = buildLayerMasks(assignPixelsToLayers(depth, layers), 3);
		const cutouts = masks.map((m) => buildLayerCutout(rgba, m, 3, 1));
		// Composite back-to-front: opaque alpha means the topmost covering layer wins.
		const composed = new Uint8ClampedArray(12);
		for (const c of cutouts) {
			for (let i = 0; i < 3; i++) {
				const j = i * 4;
				if (c[j + 3] === 255) composed.set([c[j], c[j + 1], c[j + 2], 255], j);
			}
		}
		expect(Array.from(composed)).toEqual(Array.from(rgba));
	});

	it('throws on mismatched lengths', () => {
		expect(() => buildLayerCutout(new Uint8ClampedArray(8), new Uint8Array([255]), 2, 1)).toThrow();
		expect(() =>
			buildLayerCutout(new Uint8ClampedArray(4), new Uint8Array([255, 0]), 2, 1)
		).toThrow();
	});
});

describe('depthToLayerMasks', () => {
	it('with no feather equals buildLayerMasks(assignPixelsToLayers(...))', () => {
		const layers = evenLayers(3);
		const depth = new Uint8Array([0, 100, 200, 50]);
		const masks = depthToLayerMasks(depth, layers);
		const expected = buildLayerMasks(assignPixelsToLayers(depth, layers), 3);
		expect(masks).toHaveLength(3);
		masks.forEach((m, i) => expect(Array.from(m)).toEqual(Array.from(expected[i])));
	});

	it('a feathered override softens its band edge but keeps masks summing to ~255', () => {
		const layers = layersFromThresholds([128]); // 2 layers
		// 4x1 strip: depth puts all in layer 0; an override claims the left half for layer 1.
		const depth = new Uint8Array([0, 0, 0, 0]);
		layers[1].overrides.push({
			source: 'sam-override',
			mask: new Uint8Array([255, 255, 0, 0]),
			edgeRadius: 1,
			edgeMode: 'feather'
		});
		const [m0, m1] = depthToLayerMasks(depth, layers, 4, 1);
		// Soft seam: at least one pixel is partial (not 0/255) in each mask.
		expect(m1.some((v) => v > 0 && v < 255)).toBe(true);
		for (let i = 0; i < 4; i++) {
			expect(m0[i] + m1[i]).toBeGreaterThanOrEqual(254);
			expect(m0[i] + m1[i]).toBeLessThanOrEqual(255);
		}
	});

	it('a feathered subtract override softly moves the region to the layer behind', () => {
		const layers = layersFromThresholds([128]); // 2 layers
		// 4x1: depth puts all pixels in layer 1 (near). Subtract the left half (with feather)
		// from layer 1 → it moves to layer 0, with a soft seam in the middle.
		const depth = new Uint8Array([200, 200, 200, 200]);
		layers[1].overrides.push({
			source: 'sam-override',
			op: 'subtract',
			mask: new Uint8Array([255, 255, 0, 0]),
			edgeRadius: 1,
			edgeMode: 'feather'
		});
		const [m0, m1] = depthToLayerMasks(depth, layers, 4, 1);
		// Far-left fully moved to layer 0; far-right stays in layer 1; soft transition between.
		expect(m0[0]).toBe(255);
		expect(m1[0]).toBe(0);
		expect(m1[3]).toBe(255);
		expect(m0[3]).toBe(0);
		expect(m0.some((v) => v > 0 && v < 255)).toBe(true);
		for (let i = 0; i < 4; i++) expect(m0[i] + m1[i]).toBeGreaterThanOrEqual(254);
	});

	it('does not change hard pixel ownership when an override is feathered', () => {
		const layers = layersFromThresholds([128]);
		const depth = new Uint8Array([0, 0, 0, 0]);
		layers[1].overrides.push({
			source: 'sam-override',
			mask: new Uint8Array([255, 0, 0, 0]),
			edgeRadius: 2,
			edgeMode: 'feather'
		});
		// Ownership (used by cumulative mode) is unaffected by feather: pixel 0 → layer 1.
		expect(Array.from(assignPixelsToLayers(depth, layers))).toEqual([1, 0, 0, 0]);
	});
});

describe('evenThresholds', () => {
	it('returns layerCount - 1 cuts', () => {
		expect(evenThresholds(2)).toHaveLength(1);
		expect(evenThresholds(3)).toHaveLength(2);
		expect(evenThresholds(5)).toHaveLength(4);
	});

	it('matches evenLayers boundaries', () => {
		const layers = evenLayers(4);
		const cuts = evenThresholds(4);
		expect(cuts).toEqual(layers.slice(0, -1).map((l) => l.depthMax));
	});
});

describe('layersFromThresholds', () => {
	it('produces layers that partition 0..256', () => {
		const layers = layersFromThresholds([50, 150]);
		expect(layers).toHaveLength(3);
		expect(layers[0]).toMatchObject({ depthMin: 0, depthMax: 50 });
		expect(layers[1]).toMatchObject({ depthMin: 50, depthMax: 150 });
		expect(layers[2]).toMatchObject({ depthMin: 150, depthMax: 256 });
	});

	it('rejects non-ascending thresholds', () => {
		expect(() => layersFromThresholds([100, 100])).toThrow();
		expect(() => layersFromThresholds([100, 50])).toThrow();
	});

	it('works with a single threshold (2 layers)', () => {
		const layers = layersFromThresholds([128]);
		expect(layers).toHaveLength(2);
		expect(layers[0].depthMax).toBe(128);
		expect(layers[1].depthMin).toBe(128);
	});
});

describe('resizeThresholds', () => {
	it('returns empty for target < 2 layers', () => {
		expect(resizeThresholds([100], 1)).toEqual([]);
	});

	it('returns unchanged when target matches current', () => {
		expect(resizeThresholds([85, 170], 3)).toEqual([85, 170]);
	});

	it('preserves user cuts when growing', () => {
		const grown = resizeThresholds([30, 180], 4);
		expect(grown).toHaveLength(3);
		expect(grown).toContain(30);
		expect(grown).toContain(180);
	});

	it('splits the widest gap when growing', () => {
		// gaps: [0..30]=30, [30..180]=150, [180..256]=76 → widest is 150
		// new cut should be at (30+180)/2 = 105
		const grown = resizeThresholds([30, 180], 4);
		expect(grown).toContain(105);
	});

	it('shrinks by always dropping the foreground (highest) cut', () => {
		expect(resizeThresholds([30, 80, 200], 3)).toEqual([30, 80]);
		expect(resizeThresholds([51, 102, 154, 205], 4)).toEqual([51, 102, 154]);
		expect(resizeThresholds([30, 100, 150, 213], 3)).toEqual([30, 100]);
	});

	it('grow-then-shrink restores original when the new cut went to the right', () => {
		// [50, 150] → grown to 4: widest gap is [150..256], new cut at 203 → [50, 150, 203]
		// Shrunk to 3: pops 203 → [50, 150] ✓
		const grown = resizeThresholds([50, 150], 4);
		expect(resizeThresholds(grown, 3)).toEqual([50, 150]);
	});

	it("user's drag survives growth (chain: even 3 → drag → grow to 5)", () => {
		// User adjusts thresholds from even [85, 170] to dragged [30, 170], then grows to 5.
		const result = resizeThresholds([30, 170], 5);
		expect(result).toHaveLength(4);
		expect(result).toContain(30);
		expect(result).toContain(170);
	});
});

describe('depthHistogram', () => {
	it('counts pixel occurrences per depth bin', () => {
		const depth = new Uint8Array([0, 0, 128, 128, 128, 255]);
		const hist = depthHistogram(depth);
		expect(hist.length).toBe(256);
		expect(hist[0]).toBe(2);
		expect(hist[128]).toBe(3);
		expect(hist[255]).toBe(1);
		expect(hist[42]).toBe(0);
	});

	it('returns all zeros for empty depth', () => {
		const hist = depthHistogram(new Uint8Array(0));
		expect(Array.from(hist)).toEqual(Array(256).fill(0));
	});
});
