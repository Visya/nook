import type { EdgeMode, Layer, LayerOverride } from './types';

export const DEPTH_MIN = 0;
export const DEPTH_MAX = 256;

/**
 * Build `count` evenly-spaced layers covering the full depth range, back-to-front.
 * Index 0 = farthest (lowest depth values), index count-1 = nearest.
 */
export function evenLayers(count: number): Layer[] {
	if (count < 2) throw new Error('Need at least 2 layers');
	const layers: Layer[] = [];
	for (let i = 0; i < count; i++) {
		const min = i === 0 ? DEPTH_MIN : Math.round((i * DEPTH_MAX) / count);
		const max = i === count - 1 ? DEPTH_MAX : Math.round(((i + 1) * DEPTH_MAX) / count);
		layers.push({ depthMin: min, depthMax: max, overrides: [] });
	}
	return layers;
}

/** The layer a `subtract` override on layer `l` pushes its region to: behind if possible. */
export function subtractTarget(l: number, layerCount: number): number {
	return l > 0 ? l - 1 : Math.min(l + 1, layerCount - 1);
}

/**
 * Assign each pixel to a layer index based on depth + overrides.
 * Depth values come from a single-channel depth map (0..255).
 *
 * `add` overrides win over depth thresholds; when a pixel is claimed by multiple, the
 * later (more foreground) layer wins. `subtract` overrides then run: a pixel that the
 * mask covers and that is currently assigned to that layer is pushed to the layer behind.
 */
export function assignPixelsToLayers(depth: Uint8Array, layers: Layer[]): Uint8Array {
	const n = depth.length;
	const out = new Uint8Array(n);

	for (let i = 0; i < n; i++) {
		const d = depth[i];
		let layer = layers.length - 1;
		for (let l = 0; l < layers.length; l++) {
			if (d >= layers[l].depthMin && d < layers[l].depthMax) {
				layer = l;
				break;
			}
		}
		out[i] = layer;
	}

	const checkLen = (ov: LayerOverride) => {
		if (ov.mask.length !== n) {
			throw new Error(`Override mask length ${ov.mask.length} does not match depth length ${n}`);
		}
	};

	// Additive overrides force their region into the layer (later/foreground wins).
	for (let l = 0; l < layers.length; l++) {
		for (const ov of layers[l].overrides) {
			if ((ov.op ?? 'add') !== 'add') continue;
			checkLen(ov);
			for (let i = 0; i < n; i++) {
				if (ov.mask[i] === 255) out[i] = l;
			}
		}
	}

	// Subtractive overrides remove their region from the layer (pixels fall to the layer behind).
	for (let l = 0; l < layers.length; l++) {
		const target = subtractTarget(l, layers.length);
		for (const ov of layers[l].overrides) {
			if ((ov.op ?? 'add') !== 'subtract') continue;
			checkLen(ov);
			for (let i = 0; i < n; i++) {
				if (ov.mask[i] === 255 && out[i] === l) out[i] = target;
			}
		}
	}

	return out;
}

/**
 * Build cumulative B&W masks from per-pixel layer assignments.
 * Returns `layerCount - 1` masks — the frontmost layer has no mask.
 *
 * Mask k (0-indexed): pixels assigned to layers 0..k are BLACK (0),
 * pixels in layers k+1..N-1 are WHITE (255). White = "in front of this cut".
 */
export function buildCumulativeMasks(pixelLayers: Uint8Array, layerCount: number): Uint8Array[] {
	if (layerCount < 2) return [];
	const masks: Uint8Array[] = [];
	for (let k = 0; k < layerCount - 1; k++) {
		const m = new Uint8Array(pixelLayers.length);
		for (let i = 0; i < pixelLayers.length; i++) {
			m[i] = pixelLayers[i] <= k ? 0 : 255;
		}
		masks.push(m);
	}
	return masks;
}

/**
 * Convenience: depth map → cumulative masks in one call.
 */
export function depthToMasks(depth: Uint8Array, layers: Layer[]): Uint8Array[] {
	return buildCumulativeMasks(assignPixelsToLayers(depth, layers), layers.length);
}

/**
 * Build isolated (non-cumulative) B&W masks from per-pixel layer assignments.
 * Returns `layerCount` masks — one per layer, including the frontmost.
 *
 * Mask k: pixels assigned to layer k are WHITE (255), every other pixel BLACK (0).
 * Unlike `buildCumulativeMasks`, mask k does not fold in the layers behind it, so
 * each mask isolates a single band and is usable on its own.
 */
export function buildLayerMasks(pixelLayers: Uint8Array, layerCount: number): Uint8Array[] {
	if (layerCount < 1) return [];
	const masks: Uint8Array[] = [];
	for (let k = 0; k < layerCount; k++) {
		const m = new Uint8Array(pixelLayers.length);
		for (let i = 0; i < pixelLayers.length; i++) {
			m[i] = pixelLayers[i] === k ? 255 : 0;
		}
		masks.push(m);
	}
	return masks;
}

const clampIndex = (v: number, max: number): number => (v < 0 ? 0 : v > max ? max : v);

/**
 * Soften a binary mask's edge into an 8-bit alpha ramp.
 *
 * A separable box blur (two 1-D passes, window = 2·radius+1) over the 0/255 mask
 * turns each hard edge into a roughly-linear ramp spanning the radius. Out-of-bounds
 * samples replicate the border pixel, so a region touching the image edge stays fully
 * white at the frame instead of fading against it. radius ≤ 0 returns a copy unchanged.
 */
export function featherMask(
	mask: Uint8Array,
	width: number,
	height: number,
	radius: number
): Uint8Array {
	const n = width * height;
	if (mask.length !== n) {
		throw new Error(`featherMask: mask length ${mask.length} does not match ${width}×${height}`);
	}
	const r = Math.round(radius);
	if (r <= 0) return mask.slice();

	const win = 2 * r + 1;
	const tmp = new Float32Array(n);
	// Horizontal pass.
	for (let y = 0; y < height; y++) {
		const row = y * width;
		let sum = 0;
		for (let k = -r; k <= r; k++) sum += mask[row + clampIndex(k, width - 1)];
		tmp[row] = sum / win;
		for (let x = 1; x < width; x++) {
			sum +=
				mask[row + clampIndex(x + r, width - 1)] - mask[row + clampIndex(x - r - 1, width - 1)];
			tmp[row + x] = sum / win;
		}
	}
	// Vertical pass.
	const out = new Uint8Array(n);
	for (let x = 0; x < width; x++) {
		let sum = 0;
		for (let k = -r; k <= r; k++) sum += tmp[clampIndex(k, height - 1) * width + x];
		out[x] = Math.round(sum / win);
		for (let y = 1; y < height; y++) {
			sum +=
				tmp[clampIndex(y + r, height - 1) * width + x] -
				tmp[clampIndex(y - r - 1, height - 1) * width + x];
			out[y * width + x] = Math.round(sum / win);
		}
	}
	return out;
}

/**
 * Grow a binary mask's white region by `radius` pixels with **hard** edges (morphological
 * dilation, square structuring element). Unlike `featherMask` this keeps a crisp 0/255 edge
 * — use it to "choke out" a couple of pixels rather than soften.
 *
 * Implemented as a separable count-in-window (zero-padded out of bounds): a pixel becomes
 * white if any source pixel within the radius is white. radius ≤ 0 returns a copy unchanged.
 */
export function expandMask(
	mask: Uint8Array,
	width: number,
	height: number,
	radius: number
): Uint8Array {
	const n = width * height;
	if (mask.length !== n) {
		throw new Error(`expandMask: mask length ${mask.length} does not match ${width}×${height}`);
	}
	const r = Math.round(radius);
	if (r <= 0) return mask.slice();

	// Horizontal pass: count of white pixels in [x-r, x+r] (out-of-bounds counts as 0).
	const tmp = new Float32Array(n);
	for (let y = 0; y < height; y++) {
		const row = y * width;
		let sum = 0;
		for (let k = 0; k <= r && k < width; k++) sum += mask[row + k] ? 1 : 0;
		tmp[row] = sum;
		for (let x = 1; x < width; x++) {
			const add = x + r;
			const rem = x - r - 1;
			if (add < width) sum += mask[row + add] ? 1 : 0;
			if (rem >= 0) sum -= mask[row + rem] ? 1 : 0;
			tmp[row + x] = sum;
		}
	}
	// Vertical pass over the counts: white if any white fell in the square neighborhood.
	const out = new Uint8Array(n);
	for (let x = 0; x < width; x++) {
		let sum = 0;
		for (let k = 0; k <= r && k < height; k++) sum += tmp[k * width + x];
		out[x] = sum > 0 ? 255 : 0;
		for (let y = 1; y < height; y++) {
			const add = y + r;
			const rem = y - r - 1;
			if (add < height) sum += tmp[add * width + x];
			if (rem >= 0) sum -= tmp[rem * width + x];
			out[y * width + x] = sum > 0 ? 255 : 0;
		}
	}
	return out;
}

/**
 * Grow a mask edge by `radius` px, either softly (`feather`) or with a hard dilation
 * (`expand`). radius ≤ 0 returns a copy unchanged.
 */
export function applyEdge(
	mask: Uint8Array,
	width: number,
	height: number,
	radius: number,
	mode: EdgeMode
): Uint8Array {
	if (Math.round(radius) <= 0) return mask.slice();
	return mode === 'expand'
		? expandMask(mask, width, height, radius)
		: featherMask(mask, width, height, radius);
}

/**
 * Apply a single edge adjustment (feather or expand) to every mask in a set — used to
 * grow/soften the final layer masks themselves, independent of any per-object edge.
 * radius ≤ 0 returns the input array unchanged.
 */
export function applyEdgeToMasks(
	masks: Uint8Array[],
	width: number,
	height: number,
	radius: number,
	mode: EdgeMode
): Uint8Array[] {
	if (Math.round(radius) <= 0) return masks;
	return masks.map((m) => applyEdge(m, width, height, radius, mode));
}

/**
 * Depth map → isolated per-layer masks.
 *
 * When no override carries a feather radius (or dimensions are unknown) this is just
 * `buildLayerMasks(assignPixelsToLayers(...))` — hard, disjoint, summing to all-white.
 *
 * When an override has `featherRadius > 0`, that object's contribution is composited
 * as soft alpha (`featherMask`) over the depth-only assignment in foreground order, so
 * its edge falls off gradually and the band behind it shows through the soft seam.
 * Pixel ownership itself stays hard — feather is purely an output-alpha effect.
 */
export function depthToLayerMasks(
	depth: Uint8Array,
	layers: Layer[],
	width = 0,
	height = 0
): Uint8Array[] {
	const n = depth.length;
	const hasEdge = layers.some((l) => l.overrides.some((o) => (o.edgeRadius ?? 0) > 0));
	if (!hasEdge || width * height !== n) {
		return buildLayerMasks(assignPixelsToLayers(depth, layers), layers.length);
	}

	// Base layer assignment from depth alone (overrides are painted on below).
	const masks = layers.map(() => new Uint8Array(n));
	for (let i = 0; i < n; i++) {
		const d = depth[i];
		let owner = layers.length - 1;
		for (let l = 0; l < layers.length; l++) {
			if (d >= layers[l].depthMin && d < layers[l].depthMax) {
				owner = l;
				break;
			}
		}
		masks[owner][i] = 255;
	}

	const edgeAlpha = (ov: LayerOverride) => {
		if (ov.mask.length !== n) {
			throw new Error(`Override mask length ${ov.mask.length} does not match depth length ${n}`);
		}
		return (ov.edgeRadius ?? 0) > 0
			? applyEdge(ov.mask, width, height, ov.edgeRadius!, ov.edgeMode ?? 'feather')
			: ov.mask;
	};

	// Additive overrides: soft alpha-over, foreground (higher index) last so it wins ties.
	for (let l = 0; l < layers.length; l++) {
		for (const ov of layers[l].overrides) {
			if ((ov.op ?? 'add') !== 'add') continue;
			const a = edgeAlpha(ov);
			for (let i = 0; i < n; i++) {
				const av = a[i];
				if (av === 0) continue;
				if (av >= 255) {
					for (let k = 0; k < masks.length; k++) masks[k][i] = k === l ? 255 : 0;
					continue;
				}
				const inv = 255 - av;
				for (let k = 0; k < masks.length; k++) {
					if (k !== l) masks[k][i] = Math.round((masks[k][i] * inv) / 255);
				}
				masks[l][i] = Math.min(255, Math.round((masks[l][i] * inv) / 255) + av);
			}
		}
	}

	// Subtractive overrides: move layer l's alpha (scaled by the override) to the layer behind.
	for (let l = 0; l < layers.length; l++) {
		const target = subtractTarget(l, layers.length);
		if (target === l) continue;
		for (const ov of layers[l].overrides) {
			if ((ov.op ?? 'add') !== 'subtract') continue;
			const a = edgeAlpha(ov);
			for (let i = 0; i < n; i++) {
				if (a[i] === 0) continue;
				const moved = Math.round((masks[l][i] * a[i]) / 255);
				if (moved === 0) continue;
				masks[l][i] -= moved;
				masks[target][i] = Math.min(255, masks[target][i] + moved);
			}
		}
	}
	return masks;
}

/**
 * Compose a layer's source pixels with a mask into an RGBA cutout.
 *
 * `rgba` is the source image sampled at the mask resolution (length width·height·4).
 * Colour is copied straight through; the mask value becomes alpha (so a feathered
 * mask yields a soft-edged cutout).
 *
 * `invert` flips the alpha (`255 - mask`). With an **isolated** mask (white = one band)
 * the default keeps just that band — the N cutouts re-stack to the source. With a
 * **cumulative** mask (white = "in front of this cut") `invert` keeps the black side
 * instead — "this band and everything behind it" — producing a background-preserving
 * cutout with no hole where the foreground was.
 */
export function buildLayerCutout(
	rgba: Uint8ClampedArray,
	mask: Uint8Array,
	width: number,
	height: number,
	invert = false
): Uint8ClampedArray {
	const n = width * height;
	if (mask.length !== n) {
		throw new Error(
			`buildLayerCutout: mask length ${mask.length} does not match ${width}×${height}`
		);
	}
	if (rgba.length !== n * 4) {
		throw new Error(
			`buildLayerCutout: rgba length ${rgba.length} does not match ${width}×${height}`
		);
	}
	const out = new Uint8ClampedArray(n * 4);
	for (let i = 0; i < n; i++) {
		const j = i * 4;
		out[j] = rgba[j];
		out[j + 1] = rgba[j + 1];
		out[j + 2] = rgba[j + 2];
		out[j + 3] = invert ? 255 - mask[i] : mask[i];
	}
	return out;
}

/**
 * Build the initial threshold cuts for `layerCount` evenly-spaced layers.
 * Returns `layerCount - 1` cuts in 1..255.
 */
export function evenThresholds(layerCount: number): number[] {
	if (layerCount < 2) return [];
	const out: number[] = [];
	for (let i = 1; i < layerCount; i++) out.push(Math.round((i * DEPTH_MAX) / layerCount));
	return out;
}

/**
 * Construct layers from explicit threshold cuts (length = layerCount - 1).
 * Cuts must be sorted ascending and within [1, 255].
 */
export function layersFromThresholds(thresholds: number[]): Layer[] {
	const layers: Layer[] = [];
	let prev = DEPTH_MIN;
	for (const t of thresholds) {
		if (t <= prev) throw new Error('thresholds must be strictly ascending');
		layers.push({ depthMin: prev, depthMax: t, overrides: [] });
		prev = t;
	}
	layers.push({ depthMin: prev, depthMax: DEPTH_MAX, overrides: [] });
	return layers;
}

/**
 * Build a 256-bin histogram of depth values (count per depth bin).
 */
export function depthHistogram(depth: Uint8Array): Uint32Array {
	const hist = new Uint32Array(256);
	for (let i = 0; i < depth.length; i++) hist[depth[i]]++;
	return hist;
}

/**
 * Adjust an existing threshold array to a new layer count without losing the
 * user's positions where possible.
 *
 * - Growing: split the widest gap in half, repeatedly, until the array has
 *   targetCount-1 entries. Existing cuts are always preserved.
 * - Shrinking: always drop the foreground-most cut (highest depth). Predictable
 *   regardless of cut positions; matches the mental model that reducing layers
 *   "merges from the front."
 */
export function resizeThresholds(current: number[], targetCount: number): number[] {
	if (targetCount < 2) return [];
	const targetCutCount = targetCount - 1;
	const cuts = [...current];

	while (cuts.length < targetCutCount) {
		const bounds = [DEPTH_MIN, ...cuts, DEPTH_MAX];
		let widestGap = 0;
		let widestIdx = 0;
		for (let i = 0; i < bounds.length - 1; i++) {
			const gap = bounds[i + 1] - bounds[i];
			if (gap > widestGap) {
				widestGap = gap;
				widestIdx = i;
			}
		}
		const newCut = Math.round((bounds[widestIdx] + bounds[widestIdx + 1]) / 2);
		cuts.splice(widestIdx, 0, newCut);
	}

	while (cuts.length > targetCutCount) {
		cuts.pop();
	}

	return cuts;
}
