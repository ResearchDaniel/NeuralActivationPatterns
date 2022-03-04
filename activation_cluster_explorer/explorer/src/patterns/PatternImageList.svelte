<script lang="ts">
  import type { PatternForSample } from "../types";

  import PatternImage from "./PatternImage.svelte";

  export let samples: PatternForSample[];
  export let model: string;
  export let layer: number;
</script>

<div class="flex flex-wrap min-h-0 overflow-y-auto">
  {#each samples as sample, i}
    <div class:pr-2={i + 1 !== samples.length} class="pb-2">
      <PatternImage
        imagePath={`/api/get_image/${model}/${sample.fileName}`}
        {sample}
        {layer}
      />
    </div>
  {/each}
</div>
