<script lang="ts">
  import type { PatternForSample } from "../types";

  import PatternImage from "./PatternImage.svelte";

  export let samples: PatternForSample[];
  export let model: string;
  export let layer: number;
</script>

<div class="flex">
  {#each samples as sample, i}
    <div class={i + 1 !== samples.length ? "pr-2" : ""}>
      <PatternImage
        imagePath={`/api/get_image/${model}/${sample.fileName}`}
        {sample}
        {layer}
      />
    </div>
  {/each}
</div>
