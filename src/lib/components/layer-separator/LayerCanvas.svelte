<script lang="ts">
	interface Props {
		/** Source image RGBA at mask resolution (length width*height*4). */
		rgba: Uint8ClampedArray;
		/** Per-layer mask: 0..255 used as the cutout's alpha. */
		mask: Uint8Array;
		width: number;
		height: number;
		/** Use `255 - mask` as alpha (keeps the black side — background-preserving). */
		invert?: boolean;
		alt?: string;
	}
	let { rgba, mask, width, height, invert = false, alt = 'Layer cutout' }: Props = $props();

	let canvasEl: HTMLCanvasElement | undefined = $state();

	$effect(() => {
		if (!canvasEl) return;
		canvasEl.width = width;
		canvasEl.height = height;
		const ctx = canvasEl.getContext('2d');
		if (!ctx) return;
		const imageData = ctx.createImageData(width, height);
		for (let i = 0; i < mask.length; i++) {
			const j = i * 4;
			imageData.data[j] = rgba[j];
			imageData.data[j + 1] = rgba[j + 1];
			imageData.data[j + 2] = rgba[j + 2];
			imageData.data[j + 3] = invert ? 255 - mask[i] : mask[i];
		}
		ctx.putImageData(imageData, 0, 0);
	});
</script>

<!-- The checkerboard background shows through wherever the cutout is transparent. -->
<canvas bind:this={canvasEl} aria-label={alt}></canvas>

<style>
	canvas {
		width: 100%;
		height: auto;
		display: block;
		border: 2px solid #000;
		image-rendering: pixelated;
		--sq: 12px;
		background-color: #fff;
		background-image:
			linear-gradient(45deg, #ccc 25%, transparent 25%),
			linear-gradient(-45deg, #ccc 25%, transparent 25%),
			linear-gradient(45deg, transparent 75%, #ccc 75%),
			linear-gradient(-45deg, transparent 75%, #ccc 75%);
		background-size: calc(var(--sq) * 2) calc(var(--sq) * 2);
		background-position:
			0 0,
			0 var(--sq),
			var(--sq) calc(-1 * var(--sq)),
			calc(-1 * var(--sq)) 0;
	}
</style>
