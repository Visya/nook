<script lang="ts">
	import { pipeline, env, RawImage } from '@huggingface/transformers';
	import { onMount, onDestroy } from 'svelte';
	import ImageIcon from 'virtual:icons/lucide/image';
	import DownloadIcon from 'virtual:icons/lucide/download';
	import RefreshCcwIcon from 'virtual:icons/lucide/refresh-ccw';

	import CardInterface from '$lib/components/common/CardInterface.svelte';
	import Toolbar from '$lib/components/common/Toolbar.svelte';
	import ContentArea from '$lib/components/common/ContentArea.svelte';
	import SectionCard from '$lib/components/common/SectionCard.svelte';
	import StepHeader from '$lib/components/common/StepHeader.svelte';
	import ActionButton from '$lib/components/common/ActionButton.svelte';
	import LoadingProgress from '$lib/components/common/LoadingProgress.svelte';
	import ErrorDisplay from '$lib/components/common/ErrorDisplay.svelte';
	import { useWakeLock } from '$lib/wakeLock.svelte';
	import JSZip from 'jszip';

	import MaskCanvas from '$lib/components/layer-separator/MaskCanvas.svelte';
	import LayerCanvas from '$lib/components/layer-separator/LayerCanvas.svelte';
	import MaskInspector from '$lib/components/layer-separator/MaskInspector.svelte';
	import DepthHistogram from '$lib/components/layer-separator/DepthHistogram.svelte';
	import SamPicker from '$lib/components/layer-separator/SamPicker.svelte';
	import BrushPicker from '$lib/components/layer-separator/BrushPicker.svelte';
	import {
		depthToMasks,
		depthToLayerMasks,
		buildLayerCutout,
		applyEdgeToMasks,
		assignPixelsToLayers,
		depthHistogram,
		evenThresholds,
		layersFromThresholds,
		resizeThresholds
	} from '$lib/layer-separator/masks';
	import {
		grayscaleToBlob,
		grayscaleToBlobUrl,
		rgbaToBlob,
		rgbaToBlobUrl,
		downloadBlobUrl
	} from '$lib/layer-separator/canvas';
	import {
		loadSam,
		encodeImage,
		predictMask,
		type SamCore,
		type SamSession,
		type SamPoint
	} from '$lib/layer-separator/sam';
	import type { EdgeMode, LayerOverride, OverrideOp } from '$lib/layer-separator/types';
	import type { MaskMode, MasksResponse } from '$lib/layer-separator/masks.worker';

	type DepthOutput = { depth: RawImage };
	type DepthPipeline = (input: string) => Promise<DepthOutput>;
	type ProgressEvent = { status: string; progress?: number };

	const DEPTH_MODELS = [
		{
			id: 'onnx-community/depth-anything-v2-small',
			name: 'Depth Anything V2 small',
			size: '~100 MB',
			description: 'Faster, good for most images'
		},
		{
			id: 'onnx-community/depth-anything-v2-base',
			name: 'Depth Anything V2 base',
			size: '~390 MB',
			description: 'Slower, sharper edges'
		}
	];
	const MIN_LAYERS = 2;
	const MAX_LAYERS = 5;
	const MAX_FEATHER = 32;

	let selectedDepthModelId = $state(DEPTH_MODELS[0].id);
	const selectedDepthModelName = $derived(
		DEPTH_MODELS.find((m) => m.id === selectedDepthModelId)?.name ?? selectedDepthModelId
	);

	let isModelLoaded = $state(false);
	let isLoadingModel = $state(false);
	let isProcessing = $state(false);
	let modelLoadProgress = $state(0);
	let error = $state(false);
	let errorMessage = $state('');

	let originalImageUrl = $state<string | null>(null);
	let sourceFileName = $state<string>('image');

	// Source-of-truth depth data after inference.
	let depthData = $state<Uint8Array | null>(null);
	let depthW = $state(0);
	let depthH = $state(0);

	let layerCount = $state(3);
	let thresholds = $state<number[]>(evenThresholds(3));
	// Debounced copy of thresholds that drives the (expensive) mask recomputation.
	// Lets the histogram markers drag smoothly without waiting for masks to redraw on
	// every frame. Commits ~150ms after the last threshold change.
	let committedThresholds = $state<number[]>(evenThresholds(3));
	const COMMIT_DELAY_MS = 150;
	// Per-layer overrides, parallel to layers (overridesByLayer[i] applies to layer i).
	let overridesByLayer = $state<LayerOverride[][]>([[], [], []]);

	// Output mode: 'cumulative' = N-1 stacked cut masks (Photoshop layer masks);
	// 'isolated' = N standalone per-band masks (each selects one band on its own).
	let maskMode = $state<MaskMode>('cumulative');
	// Export each result as a B&W matte or a transparent RGBA cutout (both modes).
	let exportFormat = $state<'mask' | 'cutout'>('mask');
	// Edge applied to a new object selection (stamped onto the override at accept time).
	let objectEdgeRadius = $state(0);
	let objectEdgeMode = $state<EdgeMode>('feather');
	// Edge applied to the final layer masks themselves (after assignment), independent
	// of any per-object edge.
	let layerEdgeRadius = $state(0);
	let layerEdgeMode = $state<EdgeMode>('feather');
	const MAX_EDGE = MAX_FEATHER;

	// Open mask/cutout inspector (index into `masks`, or null when closed).
	let inspectIndex = $state<number | null>(null);

	// Source image sampled to the mask resolution; the colour data for RGBA cutouts.
	let sourceRgba = $state.raw<Uint8ClampedArray | null>(null);
	// Cutout export needs source pixels ready.
	const showCutouts = $derived(exportFormat === 'cutout' && sourceRgba !== null);
	// Cumulative masks are "in front of this cut", so cutting them out keeps the BLACK
	// side (this band + everything behind it) → invert the alpha. Isolated masks are
	// already "this band only", so they're used as-is.
	const cutoutInvert = $derived(maskMode === 'cumulative');

	// SAM state
	let samCore = $state<SamCore | null>(null);
	let samSession = $state<SamSession | null>(null);
	let samStatus = $state<'idle' | 'loading' | 'encoding' | 'ready' | 'error'>('idle');
	let samLoadProgress = $state(0);
	let samErrorMessage = $state('');
	let editingLayerIndex = $state<number | null>(null);
	// Whether the current edit assigns the selection to the layer or removes it.
	let editingOp = $state<OverrideOp>('add');
	// 'sam' = click-to-segment; 'brush' = freehand paint with a destination layer.
	let editingTool = $state<'sam' | 'brush'>('sam');
	// Destination layer for a brush edit (which mask the painted region is assigned to).
	let editingTarget = $state(0);
	let brushSize = $state(20);
	// Snapshot of the edited layer's current mask, shown under the brush as a guide.
	let brushReference = $state.raw<Uint8Array | null>(null);
	let pendingMask = $state<Uint8Array | null>(null);
	let pickedPoints = $state<SamPoint[]>([]);
	let isPredicting = $state(false);

	const histogram = $derived.by(() =>
		depthData ? depthHistogram(depthData) : new Uint32Array(256)
	);

	const layers = $derived.by(() => {
		const base = layersFromThresholds(committedThresholds);
		for (let i = 0; i < base.length; i++) {
			base[i].overrides = overridesByLayer[i] ?? [];
		}
		return base;
	});

	let commitTimer: ReturnType<typeof setTimeout> | null = null;
	$effect(() => {
		const snapshot = thresholds;
		if (commitTimer) clearTimeout(commitTimer);
		commitTimer = setTimeout(() => {
			committedThresholds = [...snapshot];
			commitTimer = null;
		}, COMMIT_DELAY_MS);
		return () => {
			if (commitTimer) {
				clearTimeout(commitTimer);
				commitTimer = null;
			}
		};
	});

	// Cumulative masks are recomputed in a worker so the full-resolution pixel loop
	// doesn't freeze the editor on large images — the click/commit that triggered the
	// recompute returns immediately, the "Updating masks…" indicator paints and keeps
	// animating, and the result is applied when the worker reports back.
	// $state.raw keeps the typed-array masks un-proxied (matches the old $derived).
	let masks = $state.raw<Uint8Array[]>([]);
	let isComputingMasks = $state(false);

	let masksWorker: Worker | null = null;
	let workerFailed = false;
	// Monotonic id so a slow response for stale input is ignored when newer input
	// has already been dispatched.
	let masksRequestId = 0;

	function ensureMasksWorker(): Worker | null {
		if (workerFailed) return null;
		if (!masksWorker) {
			try {
				masksWorker = new Worker(new URL('$lib/layer-separator/masks.worker.js', import.meta.url), {
					type: 'module'
				});
				masksWorker.onmessage = (e: MessageEvent<MasksResponse>) => {
					if (e.data.id !== masksRequestId) return; // stale result
					masks = e.data.masks;
					isComputingMasks = false;
				};
				masksWorker.onerror = () => {
					// Fall back to main-thread compute for the rest of the session.
					workerFailed = true;
					masksWorker = null;
					if (depthData) {
						const base =
							maskMode === 'isolated'
								? depthToLayerMasks(depthData, layers, depthW, depthH)
								: depthToMasks(depthData, layers);
						masks = applyEdgeToMasks(base, depthW, depthH, layerEdgeRadius, layerEdgeMode);
					}
					isComputingMasks = false;
				};
			} catch {
				workerFailed = true;
				return null;
			}
		}
		return masksWorker;
	}

	$effect(() => {
		const depth = depthData;
		const currentLayers = layers;
		const mode = maskMode;
		const width = depthW;
		const height = depthH;
		const edgeRadius = layerEdgeRadius;
		const edgeMode = layerEdgeMode;
		if (!depth) {
			masks = [];
			isComputingMasks = false;
			return;
		}
		isComputingMasks = true;
		const id = ++masksRequestId;
		// Plain snapshot the worker can structured-clone (overrides live in $state).
		const payloadLayers = currentLayers.map((l) => ({
			depthMin: l.depthMin,
			depthMax: l.depthMax,
			overrides: l.overrides.map((o) => ({
				source: o.source,
				mask: o.mask,
				op: o.op,
				edgeRadius: o.edgeRadius,
				edgeMode: o.edgeMode
			}))
		}));

		const compute = () => {
			const base =
				mode === 'isolated'
					? depthToLayerMasks(depth, payloadLayers, width, height)
					: depthToMasks(depth, payloadLayers);
			return applyEdgeToMasks(base, width, height, edgeRadius, edgeMode);
		};

		const worker = ensureMasksWorker();
		if (worker) {
			worker.postMessage({
				id,
				depth,
				layers: payloadLayers,
				mode,
				width,
				height,
				edgeRadius,
				edgeMode
			});
			return;
		}

		// No worker available: yield a paint so the indicator shows, then compute.
		let inner = 0;
		const outer = requestAnimationFrame(() => {
			inner = requestAnimationFrame(() => {
				if (id !== masksRequestId) return;
				masks = compute();
				isComputingMasks = false;
			});
		});
		return () => {
			cancelAnimationFrame(outer);
			if (inner) cancelAnimationFrame(inner);
		};
	});

	// True while the on-screen masks are stale relative to the user's input: during
	// the threshold-drag debounce window and while the recompute above is in flight.
	const masksUpdating = $derived(
		isComputingMasks ||
			thresholds.length !== committedThresholds.length ||
			thresholds.some((t, i) => t !== committedThresholds[i])
	);

	let depthEstimator: DepthPipeline | null = null;
	const { requestWakeLock, releaseWakeLock, setupWakeLock } = useWakeLock();

	onMount(() => {
		if (env.backends?.onnx?.wasm) {
			env.backends.onnx.wasm.wasmPaths = '/transformers/';
		}
		// Mutates the transformers.js global config; this conflicts with the background-remover
		// which sets its own remoteHost to BASE_MODEL_URL at module top-level. First visit wins
		// per session — re-visit doesn't reload the module. Switch to BASE_MODEL_URL once the
		// depth + SAM models are hosted on the project CDN.
		env.remoteHost = 'https://huggingface.co/';
		env.remotePathTemplate = '{model}/resolve/{revision}/';
		const cleanup = setupWakeLock(() => isProcessing || isLoadingModel);
		loadModel();
		return cleanup;
	});

	async function loadModel() {
		try {
			isLoadingModel = true;
			error = false;
			modelLoadProgress = 0;
			await requestWakeLock();

			depthEstimator = (await pipeline('depth-estimation', selectedDepthModelId, {
				progress_callback: (progress: ProgressEvent) => {
					if (progress.status === 'progress') {
						modelLoadProgress = Math.round(progress.progress ?? 0);
					} else if (progress.status === 'ready') {
						modelLoadProgress = 100;
					}
				}
			})) as unknown as DepthPipeline;

			modelLoadProgress = 100;
			isModelLoaded = true;
		} catch (err) {
			console.error('Model loading error:', err);
			error = true;
			errorMessage = 'Failed to load depth model. Please check your connection and try again.';
		} finally {
			isLoadingModel = false;
			await releaseWakeLock();
		}
	}

	// Draw the source image onto a canvas at the mask resolution and read back its RGBA
	// so per-layer cutouts have colour data aligned pixel-for-pixel with the masks.
	async function computeSourceRgba(
		url: string,
		w: number,
		h: number
	): Promise<Uint8ClampedArray | null> {
		try {
			const img = new Image();
			img.src = url;
			await img.decode();
			const canvas = document.createElement('canvas');
			canvas.width = w;
			canvas.height = h;
			const ctx = canvas.getContext('2d');
			if (!ctx) return null;
			ctx.drawImage(img, 0, 0, w, h);
			return ctx.getImageData(0, 0, w, h).data;
		} catch (err) {
			console.warn('Could not read source pixels for cutouts:', err);
			return null;
		}
	}

	async function processImage(imageUrl: string) {
		if (!depthEstimator) return;
		try {
			isProcessing = true;
			error = false;
			sourceRgba = null;
			await requestWakeLock();

			const out = await depthEstimator(imageUrl);
			const depthRaw = out.depth;
			depthW = depthRaw.width;
			depthH = depthRaw.height;
			depthData =
				depthRaw.data instanceof Uint8Array
					? depthRaw.data
					: new Uint8Array(depthRaw.data as ArrayLike<number>);

			sourceRgba = await computeSourceRgba(imageUrl, depthW, depthH);

			// Reset per-image state.
			thresholds = evenThresholds(layerCount);
			overridesByLayer = Array.from({ length: layerCount }, () => []);
			samSession = null;
			cancelEdit();
		} catch (err) {
			console.error('Processing error:', err);
			error = true;
			errorMessage = 'Failed to process image. Please try again.';
		} finally {
			isProcessing = false;
			await releaseWakeLock();
		}
	}

	async function ensureSamReady() {
		if (!originalImageUrl) return;
		if (samStatus === 'idle' || samStatus === 'error') {
			samStatus = 'loading';
			samLoadProgress = 0;
			samErrorMessage = '';
			try {
				if (!samCore) {
					samCore = await loadSam((pct) => (samLoadProgress = pct));
				}
				samStatus = 'encoding';
				samSession = await encodeImage(samCore, originalImageUrl);
				samStatus = 'ready';
			} catch (err) {
				console.error('SAM load/encode error:', err);
				samStatus = 'error';
				samErrorMessage =
					'Failed to load the segmentation model. Check your connection and try again.';
			}
			return;
		}
		// Already loaded, but session may be stale (different image).
		if (samStatus === 'ready' && !samSession && samCore) {
			samStatus = 'encoding';
			try {
				samSession = await encodeImage(samCore, originalImageUrl);
				samStatus = 'ready';
			} catch (err) {
				console.error('SAM encode error:', err);
				samStatus = 'error';
				samErrorMessage = 'Failed to prepare the image for segmentation.';
			}
		}
	}

	async function enterEdit(layerIndex: number, op: OverrideOp = 'add') {
		editingLayerIndex = layerIndex;
		editingTool = 'sam';
		editingOp = op;
		pendingMask = null;
		pickedPoints = [];
		await ensureSamReady();
	}

	// Isolated mask of a single layer from the current assignment — the brush guide.
	function layerReferenceMask(layerIndex: number): Uint8Array | null {
		if (!depthData) return null;
		const pixelLayers = assignPixelsToLayers(depthData, layers);
		const m = new Uint8Array(pixelLayers.length);
		for (let i = 0; i < pixelLayers.length; i++) m[i] = pixelLayers[i] === layerIndex ? 255 : 0;
		return m;
	}

	// Freehand brush edit. Needs no segmentation model — just the source image.
	function enterBrush(layerIndex: number) {
		editingLayerIndex = layerIndex;
		editingTool = 'brush';
		editingOp = 'add';
		editingTarget = layerIndex;
		brushReference = layerReferenceMask(layerIndex);
		pendingMask = null;
		pickedPoints = [];
	}

	function cancelEdit() {
		editingLayerIndex = null;
		editingTool = 'sam';
		brushReference = null;
		pendingMask = null;
		pickedPoints = [];
		isPredicting = false;
	}

	async function handleSamClick(x: number, y: number, label: 0 | 1) {
		// Ignore clicks while a prediction is in flight so points don't stack up and
		// fire overlapping predictions that race to set `pendingMask`.
		if (!samSession || editingLayerIndex === null || isPredicting) return;
		pickedPoints = [...pickedPoints, { x, y, label }];
		await runPrediction();
	}

	// Resolve after the browser has had a chance to paint (two frames: the first
	// callback runs before a paint, the second after it). Lets a loading indicator
	// become visible before a synchronous, main-thread-blocking step runs.
	function nextPaint(): Promise<void> {
		return new Promise((resolve) =>
			requestAnimationFrame(() => requestAnimationFrame(() => resolve()))
		);
	}

	async function runPrediction() {
		if (!samSession || pickedPoints.length === 0) {
			pendingMask = null;
			return;
		}
		isPredicting = true;
		// Paint the "Segmenting…" spinner before SAM inference blocks the main thread.
		await nextPaint();
		try {
			const result = await predictMask(samSession, pickedPoints);
			if (result.width !== depthW || result.height !== depthH) {
				console.warn('SAM mask resolution mismatch', result.width, result.height, depthW, depthH);
			}
			pendingMask = result.mask;
		} catch (err) {
			console.error('SAM prediction failed:', err);
			pendingMask = null;
		} finally {
			isPredicting = false;
		}
	}

	function acceptOverride() {
		if (editingLayerIndex === null || !pendingMask) return;
		// A brush stamps an additive 'paint' override onto its destination layer: the
		// painted pixels are assigned to that mask (which removes them from wherever they
		// were), so it works as a reassign — up or down — and as a same-layer touch-up.
		const targetLayer = editingTool === 'brush' ? editingTarget : editingLayerIndex;
		const override: LayerOverride = {
			source: editingTool === 'brush' ? 'paint' : 'sam-override',
			mask: pendingMask!,
			op: editingTool === 'brush' ? 'add' : editingOp,
			edgeRadius: objectEdgeRadius,
			edgeMode: objectEdgeMode
		};
		const next = overridesByLayer.map((arr, i) => (i === targetLayer ? [...arr, override] : arr));
		overridesByLayer = next;
		cancelEdit();
	}

	function handleBrushChange(mask: Uint8Array | null) {
		pendingMask = mask;
	}

	function clearPoints() {
		if (isPredicting) return;
		pickedPoints = [];
		pendingMask = null;
	}

	async function undoLastPoint() {
		if (isPredicting || pickedPoints.length === 0) return;
		pickedPoints = pickedPoints.slice(0, -1);
		await runPrediction();
	}

	function removeOverride(layerIdx: number, overrideIdx: number) {
		overridesByLayer = overridesByLayer.map((arr, i) =>
			i === layerIdx ? arr.filter((_, j) => j !== overrideIdx) : arr
		);
	}

	$effect(() => {
		if (editingLayerIndex === null) return;
		function onKeyDown(e: KeyboardEvent) {
			if ((e.metaKey || e.ctrlKey) && !e.shiftKey && e.key === 'z') {
				e.preventDefault();
				undoLastPoint();
			}
		}
		window.addEventListener('keydown', onKeyDown);
		return () => window.removeEventListener('keydown', onKeyDown);
	});

	function handleFile(e: Event) {
		const file = (e.target as HTMLInputElement).files?.[0];
		if (!file) return;
		sourceFileName = file.name.replace(/\.[^.]+$/, '');
		const url = URL.createObjectURL(file);
		if (originalImageUrl && originalImageUrl.startsWith('blob:')) {
			URL.revokeObjectURL(originalImageUrl);
		}
		originalImageUrl = url;
		processImage(url);
	}

	function setLayerCount(n: number) {
		thresholds = resizeThresholds(thresholds, n);
		// Keep overrides for layers that still exist; new layers get empty arrays.
		const next: LayerOverride[][] = [];
		for (let i = 0; i < n; i++) next.push(overridesByLayer[i] ?? []);
		overridesByLayer = next;
		layerCount = n;
		// If we were editing a layer that no longer exists, cancel.
		if (editingLayerIndex !== null && editingLayerIndex >= n) cancelEdit();
		// Keep the brush destination within range.
		if (editingTarget >= n) editingTarget = n - 1;
	}

	function onThresholdsChange(next: number[]) {
		thresholds = next;
	}

	// Download filename noun per mode + format.
	const fileNoun = $derived(
		showCutouts
			? maskMode === 'cumulative'
				? 'cutout'
				: 'layer'
			: maskMode === 'isolated'
				? 'layer'
				: 'mask'
	);

	async function downloadMask(index: number) {
		if (!depthData) return;
		if (showCutouts && sourceRgba) {
			const cutout = buildLayerCutout(sourceRgba, masks[index], depthW, depthH, cutoutInvert);
			const cutoutUrl = await rgbaToBlobUrl(cutout, depthW, depthH);
			downloadBlobUrl(cutoutUrl, `${sourceFileName}_${fileNoun}_${index + 1}.png`);
			setTimeout(() => URL.revokeObjectURL(cutoutUrl), 5000);
			return;
		}
		const url = await grayscaleToBlobUrl(masks[index], depthW, depthH);
		downloadBlobUrl(url, `${sourceFileName}_${fileNoun}_${index + 1}.png`);
		// Revoke after the click handler so the download has time to start.
		setTimeout(() => URL.revokeObjectURL(url), 5000);
	}

	async function downloadDepth() {
		if (!depthData) return;
		const url = await grayscaleToBlobUrl(depthData, depthW, depthH);
		downloadBlobUrl(url, `${sourceFileName}_depth.png`);
		setTimeout(() => URL.revokeObjectURL(url), 5000);
	}

	async function downloadAllAsZip() {
		if (masks.length === 0 || !depthData) return;
		const zip = new JSZip();
		const cutoutMode = showCutouts && sourceRgba;
		for (let i = 0; i < masks.length; i++) {
			if (cutoutMode) {
				const cutout = buildLayerCutout(sourceRgba!, masks[i], depthW, depthH, cutoutInvert);
				zip.file(
					`${sourceFileName}_${fileNoun}_${i + 1}.png`,
					await rgbaToBlob(cutout, depthW, depthH)
				);
			} else {
				const blob = await grayscaleToBlob(masks[i], depthW, depthH);
				zip.file(`${sourceFileName}_${fileNoun}_${i + 1}.png`, blob);
			}
		}
		// Also include the depth map for reference.
		const depthBlob = await grayscaleToBlob(depthData, depthW, depthH);
		zip.file(`${sourceFileName}_depth.png`, depthBlob);

		const zipBlob = await zip.generateAsync({ type: 'blob' });
		const zipUrl = URL.createObjectURL(zipBlob);
		downloadBlobUrl(zipUrl, `${sourceFileName}_layers.zip`);
		setTimeout(() => URL.revokeObjectURL(zipUrl), 5000);
	}

	function reset() {
		depthData = null;
		sourceRgba = null;
		depthW = 0;
		depthH = 0;
		if (originalImageUrl && originalImageUrl.startsWith('blob:')) {
			URL.revokeObjectURL(originalImageUrl);
		}
		originalImageUrl = null;
		thresholds = evenThresholds(layerCount);
		overridesByLayer = Array.from({ length: layerCount }, () => []);
		samSession = null;
		cancelEdit();
		error = false;
	}

	function retry() {
		error = false;
		if (!isModelLoaded) loadModel();
	}

	function handleDepthModelChange(modelId: string) {
		if (modelId === selectedDepthModelId) return;
		selectedDepthModelId = modelId;
		reset();
		isModelLoaded = false;
		depthEstimator = null;
		loadModel();
	}

	onDestroy(() => {
		if (originalImageUrl && originalImageUrl.startsWith('blob:')) {
			URL.revokeObjectURL(originalImageUrl);
		}
		masksWorker?.terminate();
	});
</script>

{#if !isModelLoaded}
	<div class="loading">
		{#if error}
			<ErrorDisplay
				message={errorMessage}
				buttonText={isLoadingModel ? 'Loading...' : 'Retry'}
				onRetry={retry}
				isRetrying={isLoadingModel}
			/>
		{:else if isLoadingModel}
			<LoadingProgress
				title="Loading Depth Model"
				progress={modelLoadProgress}
				message="Downloading Depth Anything V2 (~100 MB). Cached after first load."
			/>
		{/if}
	</div>
{:else}
	<CardInterface>
		<Toolbar modelInfo="Layer Separator ({selectedDepthModelName})" ModelIcon={ImageIcon}>
			{#if depthData}
				<ActionButton onClick={reset} variant="danger" Icon={RefreshCcwIcon}>Restart</ActionButton>
			{/if}
		</Toolbar>

		<ContentArea>
			{#if !depthData}
				<SectionCard rotation={0.2} animationDelay={0}>
					<StepHeader stepNumber={1} title="Depth Model" backgroundColor="#ff69b4" />
					<div class="model-buttons">
						{#each DEPTH_MODELS as model (model.id)}
							<button
								class="model-btn"
								class:active={selectedDepthModelId === model.id}
								onclick={() => handleDepthModelChange(model.id)}
								disabled={isLoadingModel}
							>
								<span class="model-name">{model.name}</span>
								<span class="model-size">{model.size}</span>
								<span class="model-desc">{model.description}</span>
							</button>
						{/each}
					</div>
				</SectionCard>

				<SectionCard rotation={-0.1} animationDelay={0.1}>
					<StepHeader stepNumber={2} title="Upload Image" backgroundColor="#98fb98" />
					<div class="upload">
						<label class="upload-label">
							Choose an image
							<input type="file" accept="image/*" onchange={handleFile} disabled={isProcessing} />
						</label>
						<p class="hint">
							Output is B&W masks at source resolution: N−1 cumulative masks for Photoshop, or N
							standalone per-band masks — your choice after upload.
						</p>
					</div>
				</SectionCard>
			{/if}

			{#if isProcessing}
				<div class="processing">Running depth estimation…</div>
			{/if}

			{#if error}
				<ErrorDisplay message={errorMessage} buttonText="Try Again" onRetry={retry} />
			{/if}

			{#if depthData && originalImageUrl && !isProcessing}
				<SectionCard rotation={0.2} animationDelay={0}>
					<StepHeader stepNumber={3} title="Source & Depth" />
					<div class="grid">
						<figure>
							<figcaption>Original</figcaption>
							<img src={originalImageUrl} alt="Original" />
						</figure>
						<figure>
							<figcaption>Depth map (white = near)</figcaption>
							<MaskCanvas mask={depthData} width={depthW} height={depthH} alt="Depth map" />
							<button class="link-btn" onclick={downloadDepth}>Download depth map</button>
						</figure>
					</div>
				</SectionCard>

				<SectionCard rotation={-0.2} animationDelay={0.1}>
					<StepHeader stepNumber={4} title="Layers & Thresholds" backgroundColor="#ffd93d" />

					<div class="layer-count">
						<span class="layer-count-label">Layers:</span>
						{#each Array.from({ length: MAX_LAYERS - MIN_LAYERS + 1 }, (_, i) => MIN_LAYERS + i) as n (n)}
							<button
								class="count-btn"
								class:active={layerCount === n}
								onclick={() => setLayerCount(n)}
							>
								{n}
							</button>
						{/each}
					</div>

					<DepthHistogram {histogram} {thresholds} onChange={onThresholdsChange} />
					<p class="hint">
						Drag the markers on the histogram to adjust where layers split. Each cut is the depth
						value where the layer boundary sits (0 = far, 255 = near).
					</p>
				</SectionCard>

				<SectionCard rotation={0.15} animationDelay={0.15}>
					<StepHeader stepNumber={5} title="Refine with object clicks" backgroundColor="#ff69b4" />
					<p class="hint">
						Depth gets some objects wrong (e.g. the building grouped with the foreground leaves).
						Click <strong>Add object</strong> to force a shape into a layer, or
						<strong>Remove area</strong> to push a mis-grouped region to the layer behind — both use
						a segmentation model that snaps to the object you click. Or use
						<strong>Brush mask</strong> to paint a region freehand and reassign it to any mask, up or
						down.
					</p>

					<div class="layer-overrides">
						{#each layers as layer, i (i)}
							{@const isFar = i === 0}
							{@const isNear = i === layers.length - 1}
							<div class="layer-row" class:active={editingLayerIndex === i}>
								<div class="layer-row-label">
									<span class="layer-name">Layer {i + 1}</span>
									<span class="depth-range">{layer.depthMin}–{layer.depthMax}</span>
									{#if isFar}
										<span class="tag">farthest</span>
									{:else if isNear}
										<span class="tag">nearest</span>
									{/if}
								</div>
								<div class="layer-row-overrides">
									{#each overridesByLayer[i] ?? [] as ov, j (j)}
										<span
											class="override-chip"
											class:subtract={ov.op === 'subtract'}
											class:paint={ov.source === 'paint'}
											title={ov.source === 'paint'
												? 'painted area'
												: ov.op === 'subtract'
													? 'removed area'
													: ov.source}
										>
											{ov.source === 'paint'
												? `Brush ${j + 1}`
												: ov.op === 'subtract'
													? `Remove ${j + 1}`
													: `Object ${j + 1}`}
											<button
												class="chip-remove"
												aria-label="Delete override {j + 1} from layer {i + 1}"
												onclick={() => removeOverride(i, j)}
											>
												×
											</button>
										</span>
									{/each}
								</div>
								<div class="override-actions">
									<button
										class="add-override-btn"
										onclick={() => enterEdit(i, 'add')}
										disabled={editingLayerIndex !== null &&
											!(editingLayerIndex === i && editingOp === 'add' && editingTool === 'sam')}
									>
										{editingLayerIndex === i && editingOp === 'add' && editingTool === 'sam'
											? 'Editing…'
											: '+ Add object'}
									</button>
									<button
										class="add-override-btn subtract"
										onclick={() => enterEdit(i, 'subtract')}
										disabled={editingLayerIndex !== null &&
											!(
												editingLayerIndex === i &&
												editingOp === 'subtract' &&
												editingTool === 'sam'
											)}
									>
										{editingLayerIndex === i && editingOp === 'subtract' && editingTool === 'sam'
											? 'Editing…'
											: '− Remove area'}
									</button>
									<button
										class="add-override-btn brush"
										onclick={() => enterBrush(i)}
										disabled={editingLayerIndex !== null &&
											!(editingLayerIndex === i && editingTool === 'brush')}
									>
										{editingLayerIndex === i && editingTool === 'brush'
											? 'Editing…'
											: '✎ Brush mask'}
									</button>
								</div>
							</div>
						{/each}
					</div>

					{#if editingLayerIndex !== null && editingTool === 'brush'}
						<div class="sam-editor brush-editor">
							<p class="sam-instr">
								Paint over the part of <strong>Layer {editingLayerIndex + 1}</strong> you want to reassign,
								then pick the destination mask below. Painting into the same layer just extends it.
							</p>
							<div class="dest-row" role="radiogroup" aria-label="Destination layer">
								<span class="dest-label">Reassign to:</span>
								{#each Array.from({ length: layers.length }, (_, t) => t) as t (t)}
									<button
										class="dest-btn"
										class:active={editingTarget === t}
										role="radio"
										aria-checked={editingTarget === t}
										onclick={() => (editingTarget = t)}
									>
										L{t + 1}{#if t === editingLayerIndex}
											<span class="dest-dir">this</span>
										{:else if t < editingLayerIndex}
											<span class="dest-dir">↓ farther</span>
										{:else}
											<span class="dest-dir">↑ nearer</span>
										{/if}
									</button>
								{/each}
							</div>
							{#if originalImageUrl}
								<BrushPicker
									imageUrl={originalImageUrl}
									maskWidth={depthW}
									maskHeight={depthH}
									referenceMask={brushReference}
									onChange={handleBrushChange}
									bind:brushSize
									bind:edgeRadius={objectEdgeRadius}
									bind:edgeMode={objectEdgeMode}
									maxEdge={MAX_EDGE}
									overlayColor={editingTarget === editingLayerIndex
										? '255, 105, 180'
										: '120, 90, 255'}
								/>
							{/if}
							<div class="sam-actions">
								{#if pendingMask}
									<ActionButton onClick={acceptOverride} variant="success">
										{#if editingTarget === editingLayerIndex}
											Add to layer {editingTarget + 1}
										{:else}
											Move to layer {editingTarget + 1}
										{/if}
									</ActionButton>
								{/if}
								<button class="link-btn" onclick={cancelEdit}>Cancel</button>
							</div>
						</div>
					{:else if editingLayerIndex !== null}
						<div class="sam-editor">
							{#if samStatus === 'loading'}
								<p>Loading segmentation model… {samLoadProgress}%</p>
							{:else if samStatus === 'encoding'}
								<p>Preparing image…</p>
							{:else if samStatus === 'error'}
								<p class="sam-error">
									{samErrorMessage}
									<button onclick={() => enterEdit(editingLayerIndex!)}>Retry</button>
								</p>
							{:else if samStatus === 'ready' && originalImageUrl}
								<p class="sam-instr">
									{#if editingOp === 'subtract'}
										Click the area to <strong>remove from Layer {editingLayerIndex + 1}</strong> — those
										pixels fall to the layer behind. Each click refines the selection.
									{:else}
										Click anywhere on the image to build a mask for
										<strong>Layer {editingLayerIndex + 1}</strong>. Each click refines the previous
										result.
									{/if}
								</p>
								<SamPicker
									imageUrl={originalImageUrl}
									{pendingMask}
									maskWidth={depthW}
									maskHeight={depthH}
									points={pickedPoints}
									{isPredicting}
									onPick={handleSamClick}
									bind:edgeRadius={objectEdgeRadius}
									bind:edgeMode={objectEdgeMode}
									maxEdge={MAX_EDGE}
									overlayColor={editingOp === 'subtract' ? '255, 70, 70' : '255, 105, 180'}
								/>
								<div class="sam-actions">
									{#if pendingMask}
										<ActionButton
											onClick={acceptOverride}
											variant="success"
											disabled={isPredicting}
										>
											{#if editingOp === 'subtract'}
												Remove from layer {editingLayerIndex + 1} ({pickedPoints.length} point{pickedPoints.length ===
												1
													? ''
													: 's'})
											{:else}
												Accept ({pickedPoints.length} point{pickedPoints.length === 1 ? '' : 's'}
												→ layer {editingLayerIndex + 1})
											{/if}
										</ActionButton>
									{/if}
									{#if pickedPoints.length > 0}
										<button
											class="link-btn"
											onclick={undoLastPoint}
											disabled={isPredicting}
											title="Remove the last point (⌘Z)"
										>
											Undo last point
										</button>
										<button class="link-btn" onclick={clearPoints} disabled={isPredicting}>
											Clear points
										</button>
									{/if}
									<button class="link-btn" onclick={cancelEdit}>Cancel</button>
								</div>
							{/if}
						</div>
					{/if}
				</SectionCard>

				<SectionCard rotation={0.1} animationDelay={0.2}>
					<StepHeader
						stepNumber={6}
						title={maskMode === 'isolated' ? 'Isolated Masks' : 'Cumulative Masks'}
					/>

					<div class="mode-toggle" role="radiogroup" aria-label="Mask output mode">
						<button
							class="mode-btn"
							class:active={maskMode === 'cumulative'}
							role="radio"
							aria-checked={maskMode === 'cumulative'}
							onclick={() => (maskMode = 'cumulative')}
						>
							<span class="mode-name">Cumulative</span>
							<span class="mode-desc">N−1 stacked cut masks for Photoshop</span>
						</button>
						<button
							class="mode-btn"
							class:active={maskMode === 'isolated'}
							role="radio"
							aria-checked={maskMode === 'isolated'}
							onclick={() => (maskMode = 'isolated')}
						>
							<span class="mode-name">Isolated</span>
							<span class="mode-desc">N standalone per-band masks</span>
						</button>
					</div>

					<div class="export-toggle" role="radiogroup" aria-label="Export format">
						<span class="export-label">Export as:</span>
						<button
							class="export-btn"
							class:active={exportFormat === 'mask'}
							role="radio"
							aria-checked={exportFormat === 'mask'}
							onclick={() => (exportFormat = 'mask')}
						>
							B&amp;W masks
						</button>
						<button
							class="export-btn"
							class:active={exportFormat === 'cutout'}
							role="radio"
							aria-checked={exportFormat === 'cutout'}
							onclick={() => (exportFormat = 'cutout')}
							disabled={sourceRgba === null}
							title={sourceRgba === null ? 'Source pixels unavailable for this image' : ''}
						>
							Cut-out layers (PNG)
						</button>
					</div>

					<div class="layer-edge">
						<span class="export-label">Edge:</span>
						<div class="edge-mode" role="radiogroup" aria-label="Layer edge mode">
							<button
								class="edge-mode-btn"
								class:active={layerEdgeMode === 'feather'}
								role="radio"
								aria-checked={layerEdgeMode === 'feather'}
								onclick={() => (layerEdgeMode = 'feather')}
							>
								Feather
							</button>
							<button
								class="edge-mode-btn"
								class:active={layerEdgeMode === 'expand'}
								role="radio"
								aria-checked={layerEdgeMode === 'expand'}
								onclick={() => (layerEdgeMode = 'expand')}
							>
								Expand
							</button>
						</div>
						<input
							type="range"
							min="0"
							max={MAX_EDGE}
							step="1"
							bind:value={layerEdgeRadius}
							aria-label="Layer edge radius"
						/>
						<span class="edge-value">{layerEdgeRadius}px</span>
						<span class="hint edge-hint">
							{layerEdgeMode === 'expand'
								? 'grows every mask outward (hard edge)'
								: 'softens every mask edge'}
						</span>
					</div>

					{#if masksUpdating}
						<p class="masks-updating" role="status" aria-live="polite">
							<span class="mini-spinner" aria-hidden="true"></span>
							Updating masks…
						</p>
					{/if}
					<p class="hint">
						{#if showCutouts && maskMode === 'cumulative'}
							{masks.length} transparent PNG cut-out{masks.length === 1 ? '' : 's'}. Cut-out k keeps
							layers 1..k (the background up to that depth) and drops everything in front — no hole
							behind the foreground.
						{:else if showCutouts}
							{masks.length} transparent PNG layer{masks.length === 1 ? '' : 's'} for {layers.length}
							layers. Each holds only its own band's pixels; stack them back-to-front to rebuild the image.
						{:else if maskMode === 'isolated'}
							{masks.length} mask{masks.length === 1 ? '' : 's'} for {layers.length} layers. Each mask
							is WHITE for one band only — drop it straight onto a selection, no stacking needed.
						{:else}
							{masks.length} mask{masks.length === 1 ? '' : 's'} for {layers.length} layers. Mask k is
							BLACK where layers 1..k live; the frontmost layer has no mask.
						{/if}
					</p>
					<div class="masks-actions">
						<ActionButton
							onClick={downloadAllAsZip}
							variant="success"
							Icon={DownloadIcon}
							disabled={masksUpdating}
						>
							Download all as zip
						</ActionButton>
					</div>
					<div class="masks-grid" class:updating={masksUpdating} aria-busy={masksUpdating}>
						{#each masks as mask, i (i)}
							<figure>
								<figcaption>
									{#if showCutouts && maskMode === 'cumulative'}
										Cut-out {i + 1} — layers 1–{i + 1} (background kept)
									{:else if maskMode === 'isolated'}
										Layer {i + 1} — this band only
									{:else}
										Mask {i + 1} — covers layers 1–{i + 1}
									{/if}
								</figcaption>
								<button
									class="inspect-canvas"
									onclick={() => (inspectIndex = i)}
									title="Click to inspect closer"
									aria-label="Inspect result {i + 1}"
								>
									{#if showCutouts && sourceRgba}
										<LayerCanvas
											rgba={sourceRgba}
											{mask}
											width={depthW}
											height={depthH}
											invert={cutoutInvert}
											alt="Cut-out {i + 1}"
										/>
									{:else}
										<MaskCanvas {mask} width={depthW} height={depthH} alt="Mask {i + 1}" />
									{/if}
									<span class="inspect-badge" aria-hidden="true">Click to zoom</span>
								</button>
								<div class="figure-actions">
									<ActionButton onClick={() => downloadMask(i)} Icon={DownloadIcon}>
										{#if showCutouts && maskMode === 'cumulative'}
											Download cut-out {i + 1}
										{:else if maskMode === 'isolated'}
											Download layer {i + 1}
										{:else}
											Download mask {i + 1}
										{/if}
									</ActionButton>
									<button class="link-btn" onclick={() => (inspectIndex = i)}>Inspect</button>
								</div>
							</figure>
						{/each}
					</div>
				</SectionCard>
			{/if}
		</ContentArea>
	</CardInterface>
{/if}

{#if inspectIndex !== null && masks[inspectIndex]}
	<MaskInspector
		title={showCutouts
			? maskMode === 'cumulative'
				? `Cut-out ${inspectIndex + 1}`
				: `Layer ${inspectIndex + 1}`
			: maskMode === 'isolated'
				? `Layer ${inspectIndex + 1}`
				: `Mask ${inspectIndex + 1}`}
		mask={masks[inspectIndex]}
		width={depthW}
		height={depthH}
		rgba={sourceRgba}
		cutout={showCutouts}
		invert={cutoutInvert}
		onClose={() => (inspectIndex = null)}
	/>
{/if}

<style>
	.loading {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 2rem;
		margin: 2rem 0;
	}
	.upload {
		display: flex;
		flex-direction: column;
		gap: 1rem;
		align-items: center;
	}
	.upload-label {
		padding: 0.75rem 1.5rem;
		background: #ffd93d;
		border: 3px solid #000;
		font-weight: 700;
		cursor: pointer;
		text-transform: uppercase;
		letter-spacing: 0.5px;
		font-family: 'Space Grotesk', system-ui, sans-serif;
		box-shadow: 4px 4px 0 #000;
	}
	.upload-label:hover {
		transform: translate(-2px, -2px);
		box-shadow: 6px 6px 0 #000;
	}
	.upload-label input {
		display: none;
	}
	.hint {
		color: #555;
		max-width: 60ch;
		text-align: center;
		font-size: 0.9rem;
		line-height: 1.5;
		margin: 1rem auto;
	}
	.processing {
		text-align: center;
		padding: 2rem;
		font-weight: 600;
	}
	.grid {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 1.5rem;
		margin-top: 1rem;
	}
	.layer-count {
		display: flex;
		gap: 0.5rem;
		align-items: center;
		justify-content: center;
		flex-wrap: wrap;
		margin-bottom: 1rem;
	}
	.layer-count-label {
		font-weight: 700;
		text-transform: uppercase;
		letter-spacing: 0.5px;
		font-size: 0.875rem;
	}
	.count-btn {
		min-width: 2.5rem;
		padding: 0.4rem 0.6rem;
		background: #f0f0f0;
		border: 2px solid #000;
		font-weight: 700;
		cursor: pointer;
		font-family: inherit;
		box-shadow: 3px 3px 0 #000;
	}
	.count-btn:hover {
		transform: translate(-1px, -1px);
		box-shadow: 4px 4px 0 #000;
	}
	.count-btn.active {
		background: #ffd93d;
	}
	.masks-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
		gap: 1.5rem;
		margin-top: 1rem;
	}
	.mode-toggle {
		display: flex;
		gap: 0.75rem;
		flex-wrap: wrap;
		margin: 1rem 0;
	}
	.mode-btn {
		display: flex;
		flex-direction: column;
		gap: 0.2rem;
		padding: 0.6rem 1rem;
		background: #f0f0f0;
		border: 2px solid #000;
		cursor: pointer;
		font-family: inherit;
		box-shadow: 3px 3px 0 #000;
		text-align: left;
		flex: 1;
		min-width: 200px;
	}
	.mode-btn:hover {
		transform: translate(-1px, -1px);
		box-shadow: 4px 4px 0 #000;
	}
	.mode-btn.active {
		background: #ffd93d;
	}
	.mode-name {
		font-weight: 700;
		text-transform: uppercase;
		letter-spacing: 0.5px;
		font-size: 0.9rem;
	}
	.export-toggle {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		flex-wrap: wrap;
		margin: 0 0 1rem;
	}
	.export-label {
		font-weight: 700;
		text-transform: uppercase;
		letter-spacing: 0.5px;
		font-size: 0.8rem;
	}
	.export-btn {
		padding: 0.35rem 0.7rem;
		background: #f0f0f0;
		border: 2px solid #000;
		font-weight: 700;
		cursor: pointer;
		font-family: inherit;
		font-size: 0.8rem;
	}
	.export-btn:hover:not(:disabled) {
		background: #98fb98;
	}
	.export-btn.active {
		background: #98fb98;
	}
	.export-btn:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}
	.layer-edge {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		flex-wrap: wrap;
		margin: 0 0 1rem;
	}
	.edge-mode {
		display: flex;
	}
	.edge-mode-btn {
		padding: 0.3rem 0.6rem;
		background: #f0f0f0;
		border: 2px solid #000;
		font-weight: 700;
		font-size: 0.75rem;
		text-transform: uppercase;
		letter-spacing: 0.5px;
		cursor: pointer;
		font-family: inherit;
	}
	.edge-mode-btn + .edge-mode-btn {
		border-left: none;
	}
	.edge-mode-btn.active {
		background: #ffd93d;
	}
	.layer-edge input[type='range'] {
		width: 7rem;
		cursor: pointer;
	}
	.edge-value {
		font-family: monospace;
		font-size: 0.8rem;
		color: #555;
		min-width: 2.6rem;
	}
	.edge-hint {
		margin: 0;
		text-align: left;
	}
	.inspect-canvas {
		position: relative;
		display: block;
		padding: 0;
		border: none;
		background: none;
		cursor: zoom-in;
		font-family: inherit;
		width: 100%;
	}
	.inspect-badge {
		position: absolute;
		top: 0.4rem;
		right: 0.4rem;
		background: #000;
		color: #fff;
		font-size: 0.65rem;
		font-weight: 700;
		text-transform: uppercase;
		letter-spacing: 0.5px;
		padding: 0.15rem 0.4rem;
		opacity: 0;
		transition: opacity 0.12s ease;
		pointer-events: none;
	}
	.inspect-canvas:hover .inspect-badge,
	.inspect-canvas:focus-visible .inspect-badge {
		opacity: 1;
	}
	.figure-actions {
		display: flex;
		align-items: center;
		gap: 1rem;
		flex-wrap: wrap;
	}
	.mode-desc {
		font-size: 0.75rem;
		color: #555;
	}
	.layer-overrides {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		margin: 1rem 0;
	}
	.layer-row {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		padding: 0.5rem 0.75rem;
		border: 2px solid #000;
		background: #fff;
		flex-wrap: wrap;
	}
	.layer-row.active {
		background: #ffe5f1;
		border-color: #ff69b4;
	}
	.layer-row-label {
		font-weight: 700;
		text-transform: uppercase;
		letter-spacing: 0.5px;
		font-size: 0.875rem;
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}
	.tag {
		font-size: 0.7rem;
		color: #555;
		background: #f0f0f0;
		padding: 0.1rem 0.4rem;
		text-transform: lowercase;
		letter-spacing: 0;
		font-weight: 600;
	}
	.layer-name {
		font-weight: 700;
		text-transform: uppercase;
		letter-spacing: 0.5px;
	}
	.depth-range {
		font-size: 0.7rem;
		color: #888;
		font-weight: 400;
		text-transform: none;
		letter-spacing: 0;
		font-family: monospace;
	}
	.depth-range::before {
		content: '[';
	}
	.depth-range::after {
		content: ']';
	}
	.layer-row-overrides {
		display: flex;
		gap: 0.4rem;
		flex-wrap: wrap;
		flex: 1;
	}
	.override-chip {
		display: inline-flex;
		align-items: center;
		gap: 0.25rem;
		padding: 0.2rem 0.5rem;
		background: #ff69b4;
		color: #000;
		border: 2px solid #000;
		font-size: 0.75rem;
		font-weight: 700;
	}
	.override-chip.subtract {
		background: #ff6b6b;
	}
	.override-chip.paint {
		background: #b69bff;
	}
	.chip-remove {
		background: none;
		border: none;
		font-size: 1.1rem;
		font-weight: 700;
		cursor: pointer;
		padding: 0;
		line-height: 1;
		color: #000;
	}
	.add-override-btn {
		padding: 0.4rem 0.75rem;
		background: #98fb98;
		border: 2px solid #000;
		font-weight: 700;
		cursor: pointer;
		font-family: inherit;
		box-shadow: 3px 3px 0 #000;
		font-size: 0.8rem;
		text-transform: uppercase;
		letter-spacing: 0.5px;
	}
	.add-override-btn.subtract {
		background: #ff6b6b;
	}
	.add-override-btn.brush {
		background: #b69bff;
	}
	.override-actions {
		display: flex;
		gap: 0.4rem;
		flex-wrap: wrap;
	}
	.add-override-btn:not(:disabled):hover {
		transform: translate(-1px, -1px);
		box-shadow: 4px 4px 0 #000;
	}
	.add-override-btn:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}
	.sam-editor {
		margin-top: 1rem;
		padding: 1rem;
		border: 3px dashed #ff69b4;
		background: #fff;
	}
	.sam-instr {
		font-weight: 600;
		margin: 0 0 0.75rem;
	}
	.brush-editor {
		border-color: #7a5aff;
	}
	.dest-row {
		display: flex;
		align-items: center;
		gap: 0.4rem;
		flex-wrap: wrap;
		margin: 0 0 0.75rem;
	}
	.dest-label {
		font-weight: 700;
		text-transform: uppercase;
		letter-spacing: 0.5px;
		font-size: 0.8rem;
	}
	.dest-btn {
		display: inline-flex;
		align-items: center;
		gap: 0.3rem;
		padding: 0.3rem 0.55rem;
		background: #f0f0f0;
		border: 2px solid #000;
		font-weight: 700;
		font-size: 0.8rem;
		cursor: pointer;
		font-family: inherit;
	}
	.dest-btn.active {
		background: #b69bff;
	}
	.dest-dir {
		font-size: 0.65rem;
		font-weight: 600;
		color: #555;
		text-transform: lowercase;
	}
	.dest-btn.active .dest-dir {
		color: #000;
	}
	.sam-error {
		color: #800;
		font-weight: 600;
	}
	.sam-actions {
		display: flex;
		gap: 1rem;
		align-items: center;
		margin-top: 0.75rem;
		flex-wrap: wrap;
	}
	figure {
		margin: 0;
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}
	figcaption {
		font-weight: 700;
		text-transform: uppercase;
		letter-spacing: 0.5px;
		font-size: 0.875rem;
	}
	figure img {
		width: 100%;
		height: auto;
		display: block;
		border: 2px solid #000;
		background: #fff;
	}
	.link-btn {
		background: none;
		border: none;
		color: #00f;
		text-decoration: underline;
		cursor: pointer;
		font-size: 0.875rem;
		padding: 0;
		align-self: flex-start;
	}
	.link-btn:disabled {
		opacity: 0.4;
		cursor: not-allowed;
		text-decoration: none;
	}
	@media (prefers-reduced-motion: reduce) {
		.mini-spinner {
			animation: none;
		}
	}
	.masks-actions {
		display: flex;
		justify-content: center;
		margin: 1rem 0;
	}
	.masks-updating {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		margin: 0 0 0.5rem;
		font-weight: 700;
		font-size: 0.875rem;
		text-transform: uppercase;
		letter-spacing: 0.5px;
		color: #ff69b4;
	}
	.mini-spinner {
		width: 14px;
		height: 14px;
		border: 3px solid #000;
		border-top-color: transparent;
		border-radius: 50%;
		animation: spin 0.6s linear infinite;
	}
	.masks-grid.updating {
		opacity: 0.45;
		pointer-events: none;
		transition: opacity 0.15s ease;
	}
	@keyframes spin {
		to {
			transform: rotate(360deg);
		}
	}
	.model-buttons {
		display: flex;
		gap: 1rem;
		justify-content: center;
		flex-wrap: wrap;
	}
	.model-btn {
		display: flex;
		flex-direction: column;
		gap: 0.4rem;
		padding: 1rem 1.25rem;
		background: #f8f8f8;
		border: 3px solid #000;
		border-radius: 8px;
		cursor: pointer;
		font-family: inherit;
		box-shadow: 4px 4px 0 #000;
		min-width: 200px;
		text-align: left;
	}
	.model-btn:hover:not(:disabled) {
		transform: translate(-2px, -2px);
		box-shadow: 6px 6px 0 #000;
	}
	.model-btn.active {
		background: #ffd93d;
		transform: translate(-2px, -2px);
		box-shadow: 6px 6px 0 #000;
	}
	.model-btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
	.model-name {
		font-weight: 700;
		font-size: 0.95rem;
		text-transform: uppercase;
		letter-spacing: 0.5px;
	}
	.model-size {
		font-size: 0.75rem;
		color: #555;
		font-family: monospace;
	}
	.model-desc {
		font-size: 0.8rem;
		color: #444;
	}
	@media (max-width: 700px) {
		.grid {
			grid-template-columns: 1fr;
		}
		.layer-row {
			padding: 0.5rem;
			gap: 0.5rem;
		}
		.layer-row-label {
			width: 100%;
			justify-content: flex-start;
		}
		.add-override-btn {
			width: 100%;
		}
		.depth-range {
			display: none;
		}
	}
</style>
