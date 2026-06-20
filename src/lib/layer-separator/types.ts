export type LayerSource = 'depth-threshold' | 'sam-override' | 'paint';

export interface LayerOverride {
	source: LayerSource;
	mask: Uint8Array;
	/**
	 * Feather radius in source pixels for this object's edge. 0 / undefined = hard
	 * edge (the original behavior). Only affects isolated-masks output; the binary
	 * `mask` is kept intact so cumulative output and pixel ownership stay hard.
	 */
	featherRadius?: number;
}

export interface Layer {
	depthMin: number;
	depthMax: number;
	overrides: LayerOverride[];
}
