export type LayerSource = 'depth-threshold' | 'sam-override' | 'paint';

/**
 * How a mask edge is grown:
 * - `feather` — soft 8-bit alpha ramp (box blur).
 * - `expand` — hard dilation, grows the white region by the radius but keeps crisp edges.
 */
export type EdgeMode = 'feather' | 'expand';

/**
 * Whether an override assigns its region to the layer (`add`) or removes its region
 * from the layer (`subtract`), pushing those pixels to the layer behind it.
 */
export type OverrideOp = 'add' | 'subtract';

export interface LayerOverride {
	source: LayerSource;
	mask: Uint8Array;
	/** Add the region to the layer (default) or subtract it. */
	op?: OverrideOp;
	/**
	 * Edge radius in source pixels for this object's selection. 0 / undefined = hard
	 * edge (the original behavior). Only affects isolated-masks output; the binary
	 * `mask` is kept intact so cumulative output and pixel ownership stay hard.
	 */
	edgeRadius?: number;
	/** Whether `edgeRadius` softens (feather) or grows (expand) the edge. Default feather. */
	edgeMode?: EdgeMode;
}

export interface Layer {
	depthMin: number;
	depthMax: number;
	overrides: LayerOverride[];
}
