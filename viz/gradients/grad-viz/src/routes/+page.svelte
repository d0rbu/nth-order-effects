<script lang="ts">
    import { onMount } from 'svelte';
    import { writable } from 'svelte/store';

    const gradients = writable<any>(null);
    const selectedSample = writable(0);
    const selectedCheckpoint = writable(0);

    let unitSize: number = 0;
    let checkpoints: number[] = [];
    let localData: any = null;
    let tokenLengths: number[] = [];

    onMount(async () => {
        try {
            const res = await fetch('/gradients.json');
            if (!res.ok) throw new Error(`HTTP error! Status: ${res.status}`);
            const data = await res.json();
            gradients.set(data);
            localData = data;
            unitSize = data.gradients[0][0].length;
            checkpoints = data.checkpoint_steps;
            tokenLengths = data.gradients.map((sample: any) => sample.length);
        } catch (err) {
            console.error('Failed to load gradients.json:', err);
        }
    });

    function remapToColorRange(value: number): number {
        const v = Math.max(0, Math.min(0.04, value));
        return (v / 0.04) * 2 - 1;
    }

    function getColor(value: number): string {
        const v = remapToColorRange(value);
        const r = v > 0 ? Math.floor(v * 255) : 0;
        const b = v < 0 ? Math.floor(-v * 255) : 0;
        return `rgb(${r},0,${b})`;
    }
</script>

<h1 class="text-xl font-bold mb-4">Gradient Visualization</h1>

<div class="flex flex-col gap-4 mb-6">
    <div>
        <label class="block mb-1">Select Batch Sample</label>
        <select bind:value={$selectedSample} class="p-1 border rounded">
            {#each Array(localData?.gradients?.length || 0).fill(0).map((_, i) => i) as idx}
                <option value={idx}>{idx}</option>
            {/each}
        </select>
    </div>

    <div>
        <label class="block mb-1">Select Checkpoint Step</label>
        <input type="range" min={0} max={(checkpoints?.length || 1) - 1} bind:value={$selectedCheckpoint} step={1} class="p-1 border rounded" />
        <div class="text-sm mt-1">Checkpoint: {checkpoints[$selectedCheckpoint]}</div>
    </div>
</div>

{#if localData}
    <div class="overflow-auto max-h-[80vh] border p-2">
        <div class="mb-2 text-xs text-gray-700 flex flex-wrap">
            {#each localData.tokenized_dataset[$selectedSample] as token, tokenIdx}
                <div class="px-1 border border-gray-300 rounded bg-white mr-1 mb-1">{token}</div>
            {/each}
        </div>
        <div class="grid gap-0.5 bg-gray-100"
            style="grid-template-columns: repeat({tokenLengths[$selectedSample]}, minmax(30px, 1fr));">
            {#each Array(unitSize) as _, unitIdx}
                {#each Array(tokenLengths[$selectedSample]) as _, tokenIdx}
                    {#key `${unitIdx}-${tokenIdx}`}
                        <div class="relative w-full min-w-[30px] min-h-[30px] group"
                            style="background-color: {getColor(localData.gradients[$selectedSample][tokenIdx][unitIdx][$selectedCheckpoint])}">
                            <div class="absolute z-10 bottom-full mb-1 left-1/2 -translate-x-1/2 whitespace-nowrap text-[10px] px-1 py-0.5 bg-black text-white rounded opacity-0 group-hover:opacity-100 pointer-events-none">
                                {localData.gradients[$selectedSample][tokenIdx][unitIdx][$selectedCheckpoint].toFixed(3)}
                            </div>
                        </div>
                    {/key}
                {/each}
            {/each}
        </div>
    </div>
{:else}
    <div class="text-gray-500">Loading gradients...</div>
{/if}
