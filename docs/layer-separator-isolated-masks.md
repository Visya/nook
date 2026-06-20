# Layer Separator — Isolated-Masks Mode (design)

Status: **implemented.** This document specifies a second mask-output mode for the Layer Separator
and a feather option for object selection. It builds on the existing tool described in
[`layer-separator.md`](./layer-separator.md); read that first.

What shipped vs. this design:

- Isolated-masks mode (`maskMode: 'cumulative' | 'isolated'`), the mask-output toggle, `buildLayerMasks`,
  `featherMask`, and `depthToLayerMasks` — all as specified.
- Feather on object selection: `LayerOverride.featherRadius`, a slider in `SamPicker` with a soft live
  preview, and soft alpha-over compositing in `depthToLayerMasks`.
- Worker carries `mode` + `width`/`height`; isolated downloads are named `…_layer_k.png`.
- **Layer-cutout export** (the additive feature): an "Export as: B&W masks / Cut-out layers (PNG)"
  sub-toggle, available in **both** modes. `buildLayerCutout` composites the source pixels with a
  mask as alpha; `rgbaToBlob`/`rgbaToBlobUrl` write transparent PNGs; `LayerCanvas.svelte` previews
  them over a checkerboard. Source pixels are sampled to mask resolution once after depth runs
  (`sourceRgba`). The two modes produce different — both useful — cutouts:
  - **Isolated**: each cutout holds one band only (holes where other bands are); the N cutouts
    re-stack to the exact source. For independent planes / per-band edits.
  - **Cumulative**: cut with `invert` so each cutout keeps "this band **and everything behind it**"
    (the black side of the "in front of this cut" mask). Background-preserving, no hole behind the
    foreground — the standard stacked-document / parallax / relight workflow.
- **Deferred** (still additive follow-ups): per-layer (depth-seam) feather; matte de-fringe; a
  layered-document (PSD/AE) exporter; a reconstruct-the-stack preview.

## Motivation

Today the tool has a single, **cumulative** mask-output model:

- N layers produce **N − 1** black-and-white masks.
- Mask k means "layer k and everything behind it is black, everything in front is white."
- The masks are meant to be **stacked in Photoshop**: you apply them in order and each one cuts away
  the part of the image that belongs further back. No single mask isolates one layer — mask k folds
  in every layer behind it (layers 1..k), so it only has the intended meaning once masks 1..k−1 are
  already applied.

That is exactly right for hand-compositing, but it is the wrong shape when you want a mask **for one
specific band** on its own. If you've separated an image into background / midground / foreground,
you often just want three masks — one that selects only the background, one only the midground, one
only the foreground — without each one dragging in the bands behind it. With cumulative masks the
"midground" mask also contains the background, so you can't drop it straight onto a midground
adjustment or selection.

**Isolated-masks mode** emits exactly that: **one standalone B&W mask per layer**, where white =
"this layer's pixels only" and black = everything else. Each mask is independent — none depends on
any other being applied first.

Everything before the final mask-building step is identical between the two modes (depth estimation,
thresholding, SAM overrides, per-pixel assignment). They differ only in how the per-pixel layer
assignment is turned into output masks.

## Output contract (new mode)

For N layers numbered back-to-front (1 = farthest, N = nearest):

- The tool outputs **N standalone B&W masks** at the source image's full resolution — one per layer,
  including the frontmost (which cumulative mode leaves implicit).
- Mask k: **white** = pixels assigned to layer k, **black** = every other pixel.
- The masks are **disjoint and complete**: with hard (non-feathered) edges, every pixel is white in
  exactly one mask, so OR-ing all N masks gives a fully white image.
- Usage: drop mask k straight onto a selection/adjustment for that band — no stacking, no inverting,
  no ordering dependency.

Contrast with cumulative mode:

|                 | Cumulative (current)               | Isolated (new)                      |
| --------------- | ---------------------------------- | ----------------------------------- |
| Files emitted   | N − 1 B&W masks                    | N B&W masks                         |
| Mask k means    | layers 1..k vs. the rest           | layer k only                        |
| Frontmost layer | implicit (the remainder)           | has its own mask                    |
| Standalone?     | No — mask k assumes 1..k−1 applied | Yes — each mask isolates one band   |
| Primary use     | hand-compositing the full stack    | selecting/editing one band directly |

Mode is a user choice in the UI; the mask-building step branches on it. Everything upstream is
shared.

## Additive feature: export layers, not just masks

Separately from the mask shape, the tool can offer to export each band as an **RGBA layer cutout** —
the source pixels for that band with transparency everywhere else — instead of (or alongside) a B&W
mask. Stacking the N cutouts back-to-front reconstructs the original image. This is the natural
"give me the layers, already cut out" convenience for Figma / After Effects / parallax pipelines.

This is genuinely additive: it reuses the same per-layer assignment as isolated-masks mode and just
composites the source color through each layer's matte. It is **not** required for the isolated-masks
feature and can ship later. It is sketched in [Layer cutouts (optional)](#layer-cutouts-optional)
below so the mask-building API is designed to support it without rework.

## Feather on object selection

Today a SAM override is a hard binary mask (0/255) — its edge is exactly the pixel boundary SlimSAM
returned, which can look cut-out and aliased, especially against soft-edged subjects (foliage, hair,
ink wash). **Feather** softens the override's silhouette by ramping the mask value from 255
(interior) to 0 across a user-set radius, so the band's edge falls off gradually instead of a hard
seam.

Feather applies to both modes — it softens an isolated mask's edge and (if layer cutouts are added)
the cutout's alpha — so it is implemented as a property of the **override**, independent of output
mode.

### Where feather lives in the pipeline

Pixel **ownership** stays hard. `assignPixelsToLayers` keeps returning one integer layer index per
pixel — feather does not change which layer "wins" a pixel, so there is no ambiguity about
duplicated coverage or conflicting overrides. Feather only affects the **edge value** the owning
layer's mask paints at its boundary, and it is applied in the per-layer mask step (see below), not in
the ownership pass.

This keeps the existing, well-tested integer-assignment logic untouched and confines softness to a
separate, testable stage.

### Data model

`LayerOverride` gains an optional feather radius (pixels, at source resolution):

```ts
// src/lib/layer-separator/types.ts
export interface LayerOverride {
	source: LayerSource;
	mask: Uint8Array;
	/** Feather radius in source pixels. 0 / undefined = hard edge (current behavior). */
	featherRadius?: number;
}
```

A global default feather radius (the slider's current value) is applied to each new override at
accept time. Storing it per-override (rather than globally) lets different objects carry different
softness and keeps already-accepted overrides stable when the slider moves.

Backward compatibility: `featherRadius` absent or 0 reproduces today's hard-edged behavior exactly,
so cumulative mode and existing tests are unaffected unless a feather is explicitly set.

## Mask math changes (`masks.ts`)

Shared, unchanged: `evenLayers`, `evenThresholds`, `layersFromThresholds`, `resizeThresholds`,
`depthHistogram`, **`assignPixelsToLayers`** (still returns hard per-pixel integer ownership), and
**`buildCumulativeMasks` / `depthToMasks`** (cumulative mode stays exactly as-is).

New for isolated-masks mode:

```ts
/**
 * Per-layer isolated masks (NOT cumulative). Returns N masks, one per layer.
 * Mask k: 255 where pixelLayers[i] === k, else 0.
 */
export function buildLayerMasks(pixelLayers: Uint8Array, layerCount: number): Uint8Array[];

/**
 * Soften a binary mask's edge to an 8-bit ramp.
 * radius 0 returns the input unchanged. Implementation: signed distance from the
 * region boundary mapped through a linear (or smoothstep) falloff over `radius` px,
 * OR a separable box/Gaussian blur of the binary mask clamped to 0..255.
 * Needs width/height because feather is a 2-D neighborhood operation.
 */
export function featherMask(
	mask: Uint8Array,
	width: number,
	height: number,
	radius: number
): Uint8Array;

/** Convenience: depth map → isolated per-layer masks in one call. */
export function depthToLayerMasks(depth: Uint8Array, layers: Layer[]): Uint8Array[];
```

Feather application, per layer k:

1. `buildLayerMasks` → hard binary mask for layer k.
2. If an override on layer k carries a non-zero `featherRadius`, feather that override's contribution
   to layer k's mask by its own radius before it is merged in. (Depth-threshold boundaries between
   bands are feathered only if the layer-edge-feather follow-up below is adopted; the requested
   feature feathers object selections only.)
3. Result is layer k's 8-bit mask, ready to export as grayscale PNG.

A feathered edge means the white falls off into the black across the radius; adjacent bands' masks
can both be partially non-zero in that transition band. That is intended for soft selections. The
hard-edge invariant ("exactly one owner per pixel, masks sum to all-white") holds for `featherRadius` 0.

### Layer cutouts (optional)

If/when the additive layer-export feature lands, isolated masks are the matte:

```ts
/** Compose source RGBA with a layer's (optionally feathered) mask → RGBA cutout. */
export function buildLayerCutout(
	rgba: Uint8ClampedArray,
	mask: Uint8Array,
	width: number,
	height: number
): Uint8ClampedArray;
```

i.e. `alpha[i] = mask[i]`, color copied straight through. No change to the mask functions is needed —
the cutout is a thin compositing wrapper over the same per-layer masks.

## Worker changes (`masks.worker.ts`)

The worker currently runs `depthToMasks` and transfers back N − 1 mask buffers. It gains:

- A `maskMode: 'cumulative' | 'isolated'` field on the request message; it calls `depthToMasks` or
  `depthToLayerMasks` accordingly and transfers back N − 1 or N grayscale buffers.
- Feather computed inside the worker (it is part of building the masks), keeping the per-pixel
  neighborhood loops off the main thread for large images.
- The monotonic request-id / stale-response handling and the rAF fallback are unchanged.

(If layer cutouts are added later: an extra flag plus the source `rgba` transferred in, returning RGBA
buffers. Out of scope for the mask-mode feature.)

## Export changes (`canvas.ts`)

Isolated masks are single-channel B&W, so they reuse the existing `grayscaleToBlob` /
`grayscaleToBlobUrl` exporters unchanged — only the filenames change (e.g. `band-1.png` … `band-N.png`
vs. the cumulative `mask-1.png` …). The zip-all download (an existing open follow-up) covers both.

(Layer cutouts, if added, would need a new `rgbaToBlob` that preserves alpha; `grayscaleToBlob` forces
alpha 255.)

## UI changes

`+page.svelte` (orchestration):

- New `maskMode = $state<'cumulative' | 'isolated'>('cumulative')` toggle near the results grid. The
  `$effect` that posts to the worker includes `maskMode`; the results grid renders N − 1 or N
  `MaskCanvas` outputs accordingly, with labels reflecting the meaning ("layers 1..k" vs. "band k
  only").
- New `featherRadius = $state(0)` (e.g. 0–32 px) — the default applied to overrides accepted while
  the slider is at that value. `acceptOverride` stamps the current `featherRadius` onto the new
  `LayerOverride`.

`SamPicker.svelte` (object editor):

- A **Feather** slider (0 = hard edge) in the edit toolbar alongside zoom and the
  point/accept/clear controls. Changing it can re-render the `pendingMask` preview with the feather
  applied so the user sees the softness before accepting. Disabled while `isPredicting`, like the
  other controls.

`MaskCanvas` is reused for both modes (both are grayscale). No new canvas component is needed unless
the optional layer-cutout feature lands (which would want a checkerboard-backed RGBA canvas).

## What is reused vs. new

Reused unchanged: depth estimation, histogram + thresholding, SAM load/encode/predict, the debounced
`committedThresholds`, `assignPixelsToLayers`, `buildCumulativeMasks` / `depthToMasks`, `MaskCanvas`,
`grayscaleToBlob`, the worker's request-id/stale-response plumbing.

New: `buildLayerMasks`, `featherMask`, `depthToLayerMasks`, the `featherRadius` field + slider, the
`maskMode` toggle, and the worker `maskMode` branch. Deferred (additive): `buildLayerCutout`,
`rgbaToBlob`, an RGBA results canvas.

## Edge cases & decisions

- **Image-border feather.** A feathered region touching the image edge should not ramp to black
  against the frame. Clamp the distance transform / blur at the image border (treat out-of-bounds as
  "inside the region") so border-adjacent bands stay full-white at the frame.
- **Feather radius vs. thin objects.** A radius larger than a thin object's half-width would erode it
  to never reaching full opacity. Cap the effective interior falloff so the object's core stays at
  255; document that very large radii on thin shapes look ghosted.
- **Overlap in the soft band.** With feather > 0, adjacent bands' masks overlap in the transition
  band — intended for soft selections. Feather 0 gives exact disjoint masks that sum to all-white.
- **Performance.** Feather is O(width·height·bands-with-feather). A separable box blur (two 1-D passes)
  or a single distance transform keeps it linear-ish; runs in the worker. Only bands whose overrides
  carry a non-zero radius pay the cost.
- **Cumulative mode + feather.** Feathered overrides also soften the cumulative B&W masks. The binary
  path is unaffected when `featherRadius` is 0; exact cumulative soft-mask semantics are out of scope
  here.

## Testing plan (`masks.spec.ts`)

- `buildLayerMasks`: N layers in → N masks out; each pixel white in exactly one mask; OR of all masks
  == all-255 (disjoint + complete).
- `featherMask`: radius 0 is identity; interior stays 255; a known boundary ramps monotonically to 0
  over `radius` px; symmetric falloff on a half-plane mask; border clamping keeps edge pixels white.
- `depthToLayerMasks`: end-to-end shape/length checks; equals `buildLayerMasks(assignPixelsToLayers(…))`.
- Override feather: an override with `featherRadius > 0` produces a soft mask but does **not** change
  the integer assignment from `assignPixelsToLayers` (ownership unchanged).
- (If cutouts land) `buildLayerCutout`: alpha == mask; color copied straight; length == width*height*4;
  with feather 0, stacking the N cutouts reconstructs the source exactly.

## Open follow-ups (mode-specific)

- Layer-cutout export (the additive feature above): RGBA PNGs + a checkerboard results canvas + "zip
  all layers".
- Per-layer feather control (soften depth-threshold seams between bands, not just object selections).
- Optional matte de-fringe / color-decontamination so feathered cutout edges don't carry the
  neighboring band's color.
- Preview the reconstructed stack (OR all isolated masks / composite all cutouts) as a sanity check
  in the UI.
