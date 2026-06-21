<script lang="ts">
	import { applyEdge } from '$lib/layer-separator/masks';
	import type { EdgeMode } from '$lib/layer-separator/types';
	import ZoomInIcon from 'virtual:icons/lucide/zoom-in';
	import ZoomOutIcon from 'virtual:icons/lucide/zoom-out';
	import BrushIcon from 'virtual:icons/lucide/brush';
	import EraserIcon from 'virtual:icons/lucide/eraser';

	interface Props {
		imageUrl: string;
		maskWidth: number;
		maskHeight: number;
		/** Current mask of the layer being modified, drawn faintly as a guide. */
		referenceMask?: Uint8Array | null;
		/** Emits the raw binary brush mask (or null when nothing is painted). */
		onChange: (mask: Uint8Array | null) => void;
		/** Brush diameter in mask (source) pixels. */
		brushSize?: number;
		edgeRadius?: number;
		edgeMode?: EdgeMode;
		maxEdge?: number;
		overlayColor?: string;
	}

	let {
		imageUrl,
		maskWidth,
		maskHeight,
		referenceMask = null,
		onChange,
		brushSize = $bindable(20),
		edgeRadius = $bindable(0),
		edgeMode = $bindable<EdgeMode>('feather'),
		maxEdge = 32,
		overlayColor = '255, 105, 180'
	}: Props = $props();

	let imgEl: HTMLImageElement | undefined = $state();
	let overlayCanvasEl: HTMLCanvasElement | undefined = $state();
	let refCanvasEl: HTMLCanvasElement | undefined = $state();
	let zoom = $state(1);
	const ZOOM_LEVELS = [1, 1.5, 2, 3, 4];

	let mode = $state<'paint' | 'erase'>('paint');

	// The painted mask is a plain buffer mutated in place; `paintTick` is the reactive
	// trigger that re-renders the preview, and `count` tracks how many pixels are set so
	// we know when the selection is empty.
	// Lazily allocated so the prop dimensions are read inside a function (not captured
	// at init), and the buffer persists across strokes for in-place mutation.
	let paintedBuf: Uint8Array | null = null;
	function painted(): Uint8Array {
		return (paintedBuf ??= new Uint8Array(maskWidth * maskHeight));
	}
	let paintTick = $state(0);
	let paintedCount = $state(0);

	let drawing = false;
	let prev: { mx: number; my: number } | null = null;

	// Brush ring cursor (display-pixel position + diameter).
	let ring = $state<{ x: number; y: number; d: number } | null>(null);

	function zoomIn() {
		const i = ZOOM_LEVELS.indexOf(zoom);
		if (i < ZOOM_LEVELS.length - 1) zoom = ZOOM_LEVELS[i + 1];
	}
	function zoomOut() {
		const i = ZOOM_LEVELS.indexOf(zoom);
		if (i > 0) zoom = ZOOM_LEVELS[i - 1];
	}

	function setPixel(mx: number, my: number) {
		if (mx < 0 || my < 0 || mx >= maskWidth || my >= maskHeight) return;
		const idx = my * maskWidth + mx;
		const buf = painted();
		const cur = buf[idx];
		if (mode === 'paint') {
			if (cur !== 255) {
				buf[idx] = 255;
				paintedCount++;
			}
		} else if (cur !== 0) {
			buf[idx] = 0;
			paintedCount--;
		}
	}

	function stampDisc(cx: number, cy: number, r: number) {
		const r2 = r * r;
		const x0 = Math.floor(cx - r);
		const x1 = Math.ceil(cx + r);
		const y0 = Math.floor(cy - r);
		const y1 = Math.ceil(cy + r);
		for (let y = y0; y <= y1; y++) {
			for (let x = x0; x <= x1; x++) {
				const dx = x - cx;
				const dy = y - cy;
				if (dx * dx + dy * dy <= r2) setPixel(x, y);
			}
		}
	}

	// Stamp discs along the segment so a fast drag paints a continuous stroke.
	function stampLine(ax: number, ay: number, bx: number, by: number, r: number) {
		const dist = Math.hypot(bx - ax, by - ay);
		const steps = Math.max(1, Math.ceil(dist / Math.max(1, r / 2)));
		for (let s = 1; s <= steps; s++) {
			const t = s / steps;
			stampDisc(ax + (bx - ax) * t, ay + (by - ay) * t, r);
		}
	}

	function toMask(clientX: number, clientY: number): { mx: number; my: number } | null {
		if (!imgEl) return null;
		const rect = imgEl.getBoundingClientRect();
		const mx = Math.round(((clientX - rect.left) / rect.width) * maskWidth);
		const my = Math.round(((clientY - rect.top) / rect.height) * maskHeight);
		return { mx, my };
	}

	function commit() {
		paintTick++;
		onChange(paintedCount > 0 ? painted() : null);
	}

	function onPointerDown(e: PointerEvent) {
		if (!imgEl) return;
		imgEl.setPointerCapture(e.pointerId);
		drawing = true;
		const p = toMask(e.clientX, e.clientY);
		if (!p) return;
		stampDisc(p.mx, p.my, brushSize / 2);
		prev = p;
		commit();
	}

	function updateRing(e: PointerEvent) {
		if (!imgEl) return;
		const rect = imgEl.getBoundingClientRect();
		ring = {
			x: e.clientX - rect.left,
			y: e.clientY - rect.top,
			d: brushSize * (rect.width / maskWidth)
		};
	}

	function onPointerMove(e: PointerEvent) {
		updateRing(e);
		if (!drawing) return;
		const p = toMask(e.clientX, e.clientY);
		if (!p) return;
		if (prev) stampLine(prev.mx, prev.my, p.mx, p.my, brushSize / 2);
		else stampDisc(p.mx, p.my, brushSize / 2);
		prev = p;
		commit();
	}

	function onPointerUp(e: PointerEvent) {
		if (imgEl?.hasPointerCapture(e.pointerId)) imgEl.releasePointerCapture(e.pointerId);
		drawing = false;
		prev = null;
		commit();
	}

	function onPointerLeave() {
		ring = null;
	}

	function clearPaint() {
		painted().fill(0);
		paintedCount = 0;
		commit();
	}

	// Preview reflects the edge radius/mode so the softened or grown brush reads correctly
	// before accepting. `paintTick` is read purely to re-run this when the buffer mutates.
	const previewMask = $derived.by(() => {
		if (paintTick < 0) return null;
		if (paintedCount === 0) return null;
		const buf = painted();
		return edgeRadius > 0 ? applyEdge(buf, maskWidth, maskHeight, edgeRadius, edgeMode) : buf;
	});

	$effect(() => {
		const mask = previewMask;
		if (!overlayCanvasEl) return;
		overlayCanvasEl.width = maskWidth;
		overlayCanvasEl.height = maskHeight;
		const ctx = overlayCanvasEl.getContext('2d');
		if (!ctx) return;
		ctx.clearRect(0, 0, maskWidth, maskHeight);
		if (!mask) return;
		const imageData = ctx.createImageData(maskWidth, maskHeight);
		const [r, g, b] = overlayColor.split(',').map((s) => parseInt(s.trim(), 10));
		for (let i = 0; i < mask.length; i++) {
			if (mask[i] === 0) continue;
			const j = i * 4;
			imageData.data[j] = r;
			imageData.data[j + 1] = g;
			imageData.data[j + 2] = b;
			imageData.data[j + 3] = Math.round((mask[i] / 255) * 150);
		}
		ctx.putImageData(imageData, 0, 0);
	});

	// Faint guide showing the layer's current mask underneath the brush strokes.
	$effect(() => {
		const ref = referenceMask;
		if (!refCanvasEl) return;
		refCanvasEl.width = maskWidth;
		refCanvasEl.height = maskHeight;
		const ctx = refCanvasEl.getContext('2d');
		if (!ctx) return;
		ctx.clearRect(0, 0, maskWidth, maskHeight);
		if (!ref) return;
		const imageData = ctx.createImageData(maskWidth, maskHeight);
		for (let i = 0; i < ref.length; i++) {
			if (ref[i] === 0) continue;
			const j = i * 4;
			imageData.data[j] = 80;
			imageData.data[j + 1] = 140;
			imageData.data[j + 2] = 255;
			imageData.data[j + 3] = Math.round((ref[i] / 255) * 70);
		}
		ctx.putImageData(imageData, 0, 0);
	});
</script>

<div class="picker">
	<div class="tool-bar">
		<div class="mode" role="radiogroup" aria-label="Brush mode">
			<button
				type="button"
				class="mode-btn"
				class:active={mode === 'paint'}
				role="radio"
				aria-checked={mode === 'paint'}
				onclick={() => (mode = 'paint')}
			>
				<BrushIcon /> Paint
			</button>
			<button
				type="button"
				class="mode-btn"
				class:active={mode === 'erase'}
				role="radio"
				aria-checked={mode === 'erase'}
				onclick={() => (mode = 'erase')}
			>
				<EraserIcon /> Erase
			</button>
		</div>
		<label class="size">
			Brush
			<input
				type="range"
				min="2"
				max="80"
				step="1"
				bind:value={brushSize}
				aria-label="Brush size"
			/>
			<span class="num">{brushSize}px</span>
		</label>
		<button
			type="button"
			class="zoom-btn"
			onclick={zoomOut}
			disabled={zoom === ZOOM_LEVELS[0]}
			aria-label="Zoom out"
		>
			<ZoomOutIcon />
		</button>
		<span class="zoom-level">{zoom}×</span>
		<button
			type="button"
			class="zoom-btn"
			onclick={zoomIn}
			disabled={zoom === ZOOM_LEVELS[ZOOM_LEVELS.length - 1]}
			aria-label="Zoom in"
		>
			<ZoomInIcon />
		</button>
	</div>

	<div class="edge-control">
		<div class="edge-mode" role="radiogroup" aria-label="Brush edge mode">
			<button
				type="button"
				class="edge-mode-btn"
				class:active={edgeMode === 'feather'}
				role="radio"
				aria-checked={edgeMode === 'feather'}
				onclick={() => (edgeMode = 'feather')}
			>
				Feather
			</button>
			<button
				type="button"
				class="edge-mode-btn"
				class:active={edgeMode === 'expand'}
				role="radio"
				aria-checked={edgeMode === 'expand'}
				onclick={() => (edgeMode = 'expand')}
			>
				Expand
			</button>
		</div>
		<input
			type="range"
			min="0"
			max={maxEdge}
			step="1"
			bind:value={edgeRadius}
			aria-label="Edge radius"
		/>
		<span class="num">{edgeRadius}px</span>
	</div>

	<div class="scroll-container">
		<div class="image-wrap" style:width="{zoom * 100}%">
			<img
				bind:this={imgEl}
				src={imageUrl}
				alt="Paint over the area you want to reassign"
				draggable="false"
				onpointerdown={onPointerDown}
				onpointermove={onPointerMove}
				onpointerup={onPointerUp}
				onpointerleave={onPointerLeave}
			/>
			<canvas class="ref" bind:this={refCanvasEl} aria-hidden="true"></canvas>
			<canvas class="overlay" bind:this={overlayCanvasEl} aria-hidden="true"></canvas>
			{#if ring}
				<span
					class="ring"
					class:erase={mode === 'erase'}
					style:left="{ring.x}px"
					style:top="{ring.y}px"
					style:width="{ring.d}px"
					style:height="{ring.d}px"
				></span>
			{/if}
		</div>
	</div>

	<p class="hint">
		<strong>Drag</strong> to paint · <strong>Erase</strong> trims the selection · zoom to refine
		small areas
		{#if paintedCount > 0}
			· <button type="button" class="link-btn" onclick={clearPaint}>Clear</button>
		{/if}
	</p>
</div>

<style>
	.picker {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}
	.tool-bar,
	.edge-control {
		display: flex;
		gap: 0.4rem;
		align-items: center;
		flex-wrap: wrap;
	}
	.mode,
	.edge-mode {
		display: flex;
	}
	.mode-btn,
	.edge-mode-btn {
		display: inline-flex;
		align-items: center;
		gap: 0.25rem;
		padding: 0.3rem 0.55rem;
		background: #f0f0f0;
		border: 2px solid #000;
		font-weight: 700;
		font-size: 0.75rem;
		text-transform: uppercase;
		letter-spacing: 0.5px;
		cursor: pointer;
		font-family: inherit;
	}
	.mode-btn + .mode-btn,
	.edge-mode-btn + .edge-mode-btn {
		border-left: none;
	}
	.mode-btn.active,
	.edge-mode-btn.active {
		background: #ffd93d;
	}
	.mode-btn :global(svg),
	.edge-mode-btn :global(svg) {
		width: 0.85rem;
		height: 0.85rem;
	}
	.size {
		display: inline-flex;
		align-items: center;
		gap: 0.4rem;
		font-weight: 700;
		font-size: 0.75rem;
		text-transform: uppercase;
		letter-spacing: 0.5px;
	}
	.size input[type='range'],
	.edge-control input[type='range'] {
		width: 6rem;
		cursor: pointer;
	}
	.num {
		min-width: 2.6rem;
		font-family: monospace;
		font-size: 0.8rem;
		color: #555;
	}
	.zoom-btn {
		padding: 0.3rem 0.5rem;
		background: #f0f0f0;
		border: 2px solid #000;
		font-weight: 700;
		cursor: pointer;
		font-family: inherit;
		display: flex;
		align-items: center;
		justify-content: center;
		min-width: 2rem;
	}
	.zoom-btn:hover:not(:disabled) {
		background: #ffd93d;
	}
	.zoom-btn:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}
	.zoom-btn :global(svg) {
		width: 1rem;
		height: 1rem;
	}
	.zoom-level {
		font-family: monospace;
		font-weight: 700;
		min-width: 2.4rem;
		text-align: center;
	}
	.scroll-container {
		max-height: 80vh;
		overflow: auto;
		border: 2px solid #000;
		background: #fff;
	}
	.image-wrap {
		position: relative;
		display: block;
		width: 100%;
	}
	img {
		display: block;
		width: 100%;
		height: auto;
		cursor: crosshair;
		user-select: none;
		-webkit-user-drag: none;
		touch-action: none;
	}
	.ref,
	.overlay {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		pointer-events: none;
		image-rendering: pixelated;
	}
	.overlay {
		mix-blend-mode: multiply;
	}
	.ring {
		position: absolute;
		border: 2px solid #ff1493;
		border-radius: 50%;
		transform: translate(-50%, -50%);
		pointer-events: none;
		box-shadow: 0 0 0 1px #fff;
		mix-blend-mode: difference;
	}
	.ring.erase {
		border-style: dashed;
		border-color: #fff;
	}
	.hint {
		font-size: 0.8rem;
		color: #555;
		margin: 0;
	}
	.link-btn {
		background: none;
		border: none;
		color: #00f;
		text-decoration: underline;
		cursor: pointer;
		font-size: 0.8rem;
		padding: 0;
	}
</style>
