<script>
    import { onMount } from 'svelte';
    import { writable } from 'svelte/store';
    import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from 'shadcn-svelte/components/ui/select';
    import { Slider } from 'shadcn-svelte/components/ui/slider';

    const gradients = writable(null);
    const selectedSample = writable(0);
    const selectedCheckpoint = writable(0);

    let tokenSize = 0;
    let unitSize = 0;
    let checkpoints = [];
    let localData = null;

    onMount(async () => {
        try {
            const res = await fetch('/gradients.json');
            if (!res.ok) throw new Error(`HTTP error! Status: ${res.status}`);
            const data = await res.json();
            gradients.set(data);
            localData = data;
            tokenSize = data.gradients[0].length;
            unitSize = data.gradients[0][0].length;
            checkpoints = data.checkpoint_steps;
        } catch (err) {
            console.error('Failed to load gradients.json:', err);
        }
    });
</script>

<h1 class="text-xl font-bold mb-4">Gradient Visualization</h1>

<div class="flex flex-col gap-4 mb-6">
    <div>
        <label class="block mb-1">Select Batch Sample</label>
        <Select bind:value={$selectedSample}>
            <SelectTrigger>
                <SelectValue placeholder="Select sample" />
            </SelectTrigger>
            <SelectContent>
                {#each Array(localData?.gradients?.length || 0).fill(0).map((_, i) => i) as idx}
                    <SelectItem value={idx}>{idx}</SelectItem>
                {/each}
            </SelectContent>
        </Select>
    </div>

    <div>
        <label class="block mb-1">Select Checkpoint Step</label>
        <Slider min={0} max={(checkpoints?.length || 1) - 1} bind:value={$selectedCheckpoint} step={1} />
        <div class="text-sm mt-1">Checkpoint: {checkpoints[$selectedCheckpoint]}</div>
    </div>
</div>

{#if localData}
    <div class="grid grid-cols-[repeat(auto-fit,_minmax(2px,_1fr))] gap-0.5 bg-gray-100">
        {#each Array(unitSize) as _, unitIdx}
            {#each Array(tokenSize) as _, tokenIdx}
                {#key `${unitIdx}-${tokenIdx}`}
                    <div class="w-2 h-2"
                        style="background-color: {getColor(localData.gradients[$selectedSample][tokenIdx][unitIdx][$selectedCheckpoint])}">
                    </div>
                {/key}
            {/each}
        {/each}
    </div>
{/if}

<script>
    function getColor(value) {
        // Normalize and map to blue-red colormap
        const v = Math.max(-1, Math.min(1, value));
        const r = v > 0 ? Math.floor(v * 255) : 0;
        const b = v < 0 ? Math.floor(-v * 255) : 0;
        return `rgb(${r},0,${b})`;
    }
</script>
