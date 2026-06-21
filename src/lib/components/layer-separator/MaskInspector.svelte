<script lang="ts">
	import MaskCanvas from './MaskCanvas.svelte';
	import LayerCanvas from './LayerCanvas.svelte';
	import ZoomInIcon from 'virtual:icons/lucide/zoom-in';
	import ZoomOutIcon from 'virtual:icons/lucide/zoom-out';
	import CloseIcon from 'virtual:icons/lucide/x';

	interface Props {
		title: string;
		mask: Uint8Array;
		width: number;
		height: number;
		/** When set (and `cutout`), renders the RGBA cutout instead of the B&W mask. */
		rgba?: Uint8ClampedArray | null;
		cutout?: boolean;
		invert?: boolean;
		onClose: () => void;
	}
	let {
		title,
		mask,
		width,
		height,
		rgba = null,
		cutout = false,
		invert = false,
		onClose
	}: Props = $props();

	const ZOOM_LEVELS = [1, 2, 3, 4, 6, 8];
	let zoom = $state(2);

	function zoomIn() {
		const i = ZOOM_LEVELS.indexOf(zoom);
		if (i < ZOOM_LEVELS.length - 1) zoom = ZOOM_LEVELS[i + 1];
	}
	function zoomOut() {
		const i = ZOOM_LEVELS.indexOf(zoom);
		if (i > 0) zoom = ZOOM_LEVELS[i - 1];
	}

	function onKeydown(e: KeyboardEvent) {
		if (e.key === 'Escape') onClose();
	}
</script>

<svelte:window onkeydown={onKeydown} />

<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
<div class="backdrop" onclick={onClose}>
	<div
		class="dialog"
		role="dialog"
		aria-modal="true"
		aria-label={title}
		tabindex="-1"
		onclick={(e) => e.stopPropagation()}
	>
		<div class="bar">
			<span class="title">{title}</span>
			<div class="zoom">
				<button
					class="btn"
					onclick={zoomOut}
					disabled={zoom === ZOOM_LEVELS[0]}
					aria-label="Zoom out"
				>
					<ZoomOutIcon />
				</button>
				<span class="level">{zoom}×</span>
				<button
					class="btn"
					onclick={zoomIn}
					disabled={zoom === ZOOM_LEVELS[ZOOM_LEVELS.length - 1]}
					aria-label="Zoom in"
				>
					<ZoomInIcon />
				</button>
				<button class="btn close" onclick={onClose} aria-label="Close">
					<CloseIcon />
				</button>
			</div>
		</div>
		<div class="scroll">
			<div class="wrap" style:width="{zoom * 100}%">
				{#if cutout && rgba}
					<LayerCanvas {rgba} {mask} {width} {height} {invert} alt={title} />
				{:else}
					<MaskCanvas {mask} {width} {height} alt={title} />
				{/if}
			</div>
		</div>
	</div>
</div>

<style>
	.backdrop {
		position: fixed;
		inset: 0;
		background: rgba(0, 0, 0, 0.7);
		display: flex;
		align-items: center;
		justify-content: center;
		z-index: 1000;
		padding: 1rem;
	}
	.dialog {
		background: #fff;
		border: 3px solid #000;
		box-shadow: 8px 8px 0 #000;
		max-width: 95vw;
		max-height: 90vh;
		display: flex;
		flex-direction: column;
		width: min(900px, 95vw);
	}
	.bar {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 1rem;
		padding: 0.6rem 0.8rem;
		border-bottom: 3px solid #000;
		background: #ffd93d;
	}
	.title {
		font-weight: 700;
		text-transform: uppercase;
		letter-spacing: 0.5px;
		font-size: 0.9rem;
	}
	.zoom {
		display: flex;
		align-items: center;
		gap: 0.3rem;
	}
	.level {
		font-family: monospace;
		font-weight: 700;
		min-width: 2.5rem;
		text-align: center;
	}
	.btn {
		display: flex;
		align-items: center;
		justify-content: center;
		padding: 0.3rem 0.5rem;
		background: #fff;
		border: 2px solid #000;
		cursor: pointer;
		font-family: inherit;
	}
	.btn:hover:not(:disabled) {
		background: #98fb98;
	}
	.btn:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}
	.btn.close:hover {
		background: #ff6b6b;
	}
	.btn :global(svg) {
		width: 1rem;
		height: 1rem;
	}
	.scroll {
		overflow: auto;
		background: #f0f0f0;
		padding: 1rem;
	}
	.wrap {
		margin: 0 auto;
	}
</style>
