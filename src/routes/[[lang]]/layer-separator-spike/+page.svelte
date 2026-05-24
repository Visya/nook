<script lang="ts">
	import { pipeline, env, RawImage } from '@huggingface/transformers';
	import { onMount } from 'svelte';

	type DepthOutput = { depth: RawImage; predicted_depth: unknown };
	type DepthPipeline = (input: string) => Promise<DepthOutput>;
	type ProgressEvent = { status: string; progress?: number };

	const MODEL_ID = 'onnx-community/depth-anything-v2-small';
	const EXAMPLE_URL = '/spike-landscape.jpg';

	let isModelLoaded = $state(false);
	let isLoadingModel = $state(false);
	let loadProgress = $state(0);
	let isProcessing = $state(false);
	let error = $state<string | null>(null);
	let depthMapUrl = $state<string | null>(null);
	let originalImageUrl = $state<string | null>(null);
	let processingMs = $state(0);
	let imageDims = $state<{ w: number; h: number } | null>(null);
	let depthDims = $state<{ w: number; h: number } | null>(null);

	let depthEstimator: DepthPipeline | null = null;

	onMount(() => {
		if (env.backends?.onnx?.wasm) {
			env.backends.onnx.wasm.wasmPaths = '/transformers/';
		}
		// Load from HF directly for the spike — no CDN setup needed yet.
		env.remoteHost = 'https://huggingface.co/';
		env.remotePathTemplate = '{model}/resolve/{revision}/';
		loadModel();
	});

	async function loadModel() {
		try {
			isLoadingModel = true;
			error = null;
			loadProgress = 0;
			depthEstimator = (await pipeline('depth-estimation', MODEL_ID, {
				progress_callback: (progress: ProgressEvent) => {
					if (progress.status === 'progress') {
						loadProgress = Math.round(progress.progress ?? 0);
					} else if (progress.status === 'ready') {
						loadProgress = 100;
					}
				}
			})) as unknown as DepthPipeline;
			isModelLoaded = true;
		} catch (err) {
			console.error('Model load failed', err);
			error = `Failed to load model: ${err instanceof Error ? err.message : String(err)}`;
		} finally {
			isLoadingModel = false;
		}
	}

	async function runDepth(imageUrl: string) {
		if (!depthEstimator) return;
		isProcessing = true;
		error = null;
		const start = performance.now();

		if (originalImageUrl && originalImageUrl.startsWith('blob:')) {
			URL.revokeObjectURL(originalImageUrl);
		}
		originalImageUrl = imageUrl;

		try {
			const out = await depthEstimator(imageUrl);
			const depthRaw = out.depth;
			depthDims = { w: depthRaw.width, h: depthRaw.height };

			const canvas = depthRaw.toCanvas();
			await new Promise<void>((resolve) => {
				canvas.toBlob((blob) => {
					if (blob) {
						if (depthMapUrl) URL.revokeObjectURL(depthMapUrl);
						depthMapUrl = URL.createObjectURL(blob);
					}
					resolve();
				}, 'image/png');
			});

			processingMs = Math.round(performance.now() - start);
		} catch (err) {
			console.error('Inference failed', err);
			error = `Inference failed: ${err instanceof Error ? err.message : String(err)}`;
		} finally {
			isProcessing = false;
		}
	}

	function handleFile(e: Event) {
		const file = (e.target as HTMLInputElement).files?.[0];
		if (!file) return;
		const url = URL.createObjectURL(file);
		runDepth(url);
	}

	function useExample() {
		runDepth(EXAMPLE_URL);
	}

	function onOriginalLoad(e: Event) {
		const img = e.target as HTMLImageElement;
		imageDims = { w: img.naturalWidth, h: img.naturalHeight };
	}
</script>

<div class="spike">
	<h1>Layer Separator — Depth Spike</h1>
	<p class="lead">
		Phase 1 throwaway page. Loads <code>{MODEL_ID}</code> via transformers.js and renders the raw depth
		map. Goal: eyeball whether Depth Anything V2 produces usable output on stylized ink landscapes before
		building the rest of the feature.
	</p>

	{#if isLoadingModel}
		<div class="status">Loading model… {loadProgress}%</div>
	{:else if !isModelLoaded}
		<div class="status error">{error ?? 'Model not loaded.'}</div>
		<button onclick={loadModel}>Retry</button>
	{:else}
		<div class="controls">
			<button onclick={useExample} disabled={isProcessing}>Run on example (ink landscape)</button>
			<label class="file-btn">
				Upload your own
				<input type="file" accept="image/*" onchange={handleFile} disabled={isProcessing} />
			</label>
		</div>

		{#if isProcessing}
			<div class="status">Running depth estimation…</div>
		{/if}

		{#if error}
			<div class="status error">{error}</div>
		{/if}

		{#if originalImageUrl || depthMapUrl}
			<div class="grid">
				<figure>
					<figcaption>Original</figcaption>
					{#if originalImageUrl}
						<img src={originalImageUrl} alt="Original" onload={onOriginalLoad} />
						{#if imageDims}
							<small>{imageDims.w} × {imageDims.h}</small>
						{/if}
					{/if}
				</figure>
				<figure>
					<figcaption>Depth map (white = near, black = far)</figcaption>
					{#if depthMapUrl}
						<img src={depthMapUrl} alt="Depth map" />
						{#if depthDims}
							<small>{depthDims.w} × {depthDims.h} · {processingMs}ms</small>
						{/if}
					{:else if !isProcessing}
						<div class="placeholder">No depth map yet.</div>
					{/if}
				</figure>
			</div>
		{/if}
	{/if}
</div>

<style>
	.spike {
		max-width: 1100px;
		margin: 2rem auto;
		padding: 1rem;
		font-family: system-ui, sans-serif;
	}
	h1 {
		margin-top: 0;
	}
	.lead {
		color: #555;
		max-width: 70ch;
	}
	code {
		background: #f0f0f0;
		padding: 0.1rem 0.3rem;
		border-radius: 3px;
		font-size: 0.9em;
	}
	.status {
		margin: 1rem 0;
		padding: 0.75rem 1rem;
		background: #eef;
		border-left: 4px solid #88a;
	}
	.status.error {
		background: #fee;
		border-left-color: #c44;
		color: #800;
	}
	.controls {
		display: flex;
		gap: 0.75rem;
		flex-wrap: wrap;
		margin: 1rem 0;
	}
	button,
	.file-btn {
		padding: 0.6rem 1rem;
		border: 2px solid #000;
		background: #ffd93d;
		font-weight: 600;
		cursor: pointer;
		font-family: inherit;
	}
	button:disabled,
	.file-btn:has(input:disabled) {
		opacity: 0.5;
		cursor: not-allowed;
	}
	.file-btn input {
		display: none;
	}
	.grid {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 1.5rem;
		margin-top: 1.5rem;
	}
	figure {
		margin: 0;
	}
	figcaption {
		font-weight: 600;
		margin-bottom: 0.5rem;
	}
	figure img {
		width: 100%;
		height: auto;
		display: block;
		border: 1px solid #ccc;
	}
	figure small {
		display: block;
		color: #777;
		margin-top: 0.25rem;
		font-size: 0.85rem;
	}
	.placeholder {
		padding: 2rem;
		background: #f5f5f5;
		text-align: center;
		color: #999;
		border: 1px dashed #ccc;
	}
	@media (max-width: 700px) {
		.grid {
			grid-template-columns: 1fr;
		}
	}
</style>
