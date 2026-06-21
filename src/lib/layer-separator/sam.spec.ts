import { describe, it, expect, vi } from 'vitest';

// Mock the transformers SDK so we don't need WASM / ONNX in the test runner.
// Only the Tensor constructor is read from the module by predictMask; the
// rest is exercised through the fake session.
vi.mock('@huggingface/transformers', () => ({
	SamModel: { from_pretrained: vi.fn() },
	AutoProcessor: { from_pretrained: vi.fn() },
	RawImage: { read: vi.fn() },
	Tensor: class {
		type: string;
		data: ArrayLike<number> | BigInt64Array | Float32Array;
		dims: number[];
		constructor(
			type: string,
			data: ArrayLike<number> | BigInt64Array | Float32Array,
			dims: number[]
		) {
			this.type = type;
			this.data = data;
			this.dims = dims;
		}
	}
}));

import { predictMask, chooseBestMask, type SamSession } from './sam';

/**
 * Build a fake SamSession that returns a fabricated mask tensor with the
 * given dims and data, and a fabricated iou_scores vector. Lets us prove the
 * NCHW indexing in predictMask without running the real model.
 */
function makeFakeSession(opts: {
	dims: number[];
	maskData: Uint8Array;
	scores: number[];
	origSize?: [number, number];
	reshapedSize?: [number, number];
}): SamSession {
	const { dims, maskData, scores, origSize = [2, 2], reshapedSize = [2, 2] } = opts;
	const tensor = { dims, data: maskData };
	return {
		core: {
			model: (async () => ({
				pred_masks: { dims: [1, 1, ...dims] },
				iou_scores: { data: new Float32Array(scores) }
			})) as unknown as SamSession['core']['model'],
			processor: {
				post_process_masks: async () => [tensor]
			}
		},
		processed: {
			original_sizes: [origSize],
			reshaped_input_sizes: [reshapedSize]
		},
		embeddings: {},
		width: origSize[1],
		height: origSize[0]
	} as unknown as SamSession;
}

describe('predictMask', () => {
	it('reads the best mask from planar NCHW data (mask 1 wins)', async () => {
		// dims = [1, 3, 2, 2] → 3 candidate masks of 2x2 pixels each.
		// Memory layout NCHW: mask 0 first 4 bytes, mask 1 next 4, mask 2 last 4.
		const maskData = new Uint8Array([
			0,
			0,
			0,
			0, // mask 0 — empty
			1,
			1,
			1,
			1, // mask 1 — all foreground
			0,
			0,
			0,
			0 // mask 2 — empty
		]);
		const session = makeFakeSession({
			dims: [1, 3, 2, 2],
			maskData,
			scores: [0.1, 0.9, 0.2]
		});
		const result = await predictMask(session, [{ x: 0, y: 0, label: 1 }]);
		expect(result.width).toBe(2);
		expect(result.height).toBe(2);
		expect(Array.from(result.mask)).toEqual([255, 255, 255, 255]);
		expect(result.score).toBeCloseTo(0.9);
	});

	it('picks mask 2 when its IOU is highest', async () => {
		const maskData = new Uint8Array([
			0,
			0,
			0,
			0, // mask 0
			0,
			0,
			0,
			0, // mask 1
			1,
			0,
			1,
			0 // mask 2
		]);
		const session = makeFakeSession({
			dims: [1, 3, 2, 2],
			maskData,
			scores: [0.1, 0.2, 0.9]
		});
		const result = await predictMask(session, [{ x: 0, y: 0, label: 1 }]);
		expect(Array.from(result.mask)).toEqual([255, 0, 255, 0]);
	});

	it('honors a subtract point over raw IOU (shift+click carves)', async () => {
		// 2x2. mask 0 covers everything (incl. the subtract point) with high IOU;
		// mask 1 excludes the bottom-right subtract point with lower IOU.
		const maskData = new Uint8Array([
			1,
			1,
			1,
			1, // mask 0 — includes the negative point at idx 3
			1,
			1,
			1,
			0, // mask 1 — excludes the negative point at idx 3
			0,
			0,
			0,
			0 // mask 2
		]);
		const session = makeFakeSession({
			dims: [1, 3, 2, 2],
			maskData,
			scores: [0.9, 0.2, 0.1]
		});
		const result = await predictMask(session, [
			{ x: 0, y: 0, label: 1 }, // keep top-left
			{ x: 1, y: 1, label: 0 } // remove bottom-right
		]);
		// Picks mask 1 (satisfies both points) even though mask 0 has the higher IOU.
		expect(Array.from(result.mask)).toEqual([255, 255, 255, 0]);
	});

	it('rejects an empty points array', async () => {
		const session = makeFakeSession({
			dims: [1, 3, 2, 2],
			maskData: new Uint8Array(12),
			scores: [0, 0, 0]
		});
		await expect(predictMask(session, [])).rejects.toThrow();
	});

	it('reads width/height from the last two dims (works for 3D tensors too)', async () => {
		// Some transformers.js versions strip the batch dim → dims [3, H, W].
		const maskData = new Uint8Array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]);
		const session = makeFakeSession({
			dims: [3, 2, 2],
			maskData,
			scores: [0, 1, 0]
		});
		const result = await predictMask(session, [{ x: 0, y: 0, label: 1 }]);
		expect(result.width).toBe(2);
		expect(result.height).toBe(2);
		expect(Array.from(result.mask)).toEqual([255, 255, 255, 255]);
	});
});

describe('chooseBestMask', () => {
	// Two 2x2 candidate masks, planar NCHW.
	const data = new Uint8Array([
		1,
		1,
		1,
		1, // mask 0 — full
		1,
		0,
		0,
		0 // mask 1 — only top-left
	]);

	it('prefers the candidate that satisfies the most points, ignoring IOU', () => {
		// Positive top-left + negative bottom-right. Mask 0 fails the negative; mask 1 passes both.
		const points: { x: number; y: number; label: 0 | 1 }[] = [
			{ x: 0, y: 0, label: 1 },
			{ x: 1, y: 1, label: 0 }
		];
		// Give mask 0 the higher IOU to prove satisfaction wins.
		expect(chooseBestMask(data, 2, 2, 2, points, [0.9, 0.1])).toBe(1);
	});

	it('breaks ties by IOU when satisfaction is equal', () => {
		// A single positive at top-left: both masks are white there → tie → higher IOU wins.
		const points: { x: number; y: number; label: 0 | 1 }[] = [{ x: 0, y: 0, label: 1 }];
		expect(chooseBestMask(data, 2, 2, 2, points, [0.3, 0.8])).toBe(1);
		expect(chooseBestMask(data, 2, 2, 2, points, [0.8, 0.3])).toBe(0);
	});

	it('clamps out-of-range point coordinates', () => {
		const points: { x: number; y: number; label: 0 | 1 }[] = [{ x: 99, y: 99, label: 0 }];
		// Bottom-right (clamped) is black in mask 1, white in mask 0 → mask 1 satisfies the subtract.
		expect(chooseBestMask(data, 2, 2, 2, points, [0.9, 0.1])).toBe(1);
	});
});
