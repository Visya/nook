import { depthToMasks, depthToLayerMasks } from './masks';
import type { Layer } from './types';

export type MaskMode = 'cumulative' | 'isolated';

export interface MasksRequest {
	id: number;
	depth: Uint8Array;
	layers: Layer[];
	mode: MaskMode;
	width: number;
	height: number;
}

export interface MasksResponse {
	id: number;
	masks: Uint8Array[];
}

// Runs the full-resolution pixel loop (and any feather blur) off the main thread so
// the editor stays responsive — and the "Updating masks…" indicator keeps animating —
// on large images.
self.onmessage = (e: MessageEvent) => {
	const { id, depth, layers, mode, width, height } = e.data as MasksRequest;
	const masks =
		mode === 'isolated'
			? depthToLayerMasks(depth, layers, width, height)
			: depthToMasks(depth, layers);
	// Transfer the mask buffers back instead of cloning them.
	const transfer = masks.map((m) => m.buffer);
	(self as unknown as Worker).postMessage({ id, masks } satisfies MasksResponse, transfer);
};
