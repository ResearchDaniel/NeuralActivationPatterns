<script lang="ts">
  import ImageTooltip from "./components/ImageTooltip.svelte";

  import Distribution from "./Distribution.svelte";
  import Patterns from "./patterns/Patterns.svelte";

  import type { PatternForSample } from "./types";

  export let model: string;
  export let layer: string;
  export let labels: string[] | number[];

  $: fetchPatterns = (async () => {
    const response = await fetch(`/api/get_patterns/${model}/${layer}`);
    const jsonResponse = await response.json();
    const patterns = JSON.parse(jsonResponse) as PatternForSample[];
    return patterns;
  })();
</script>

<div class="flex flex-col flex-grow p-2 min-h-0">
  {#await fetchPatterns then patterns}
    <Distribution {patterns} {labels} />
    <Patterns {model} {layer} {patterns} />
    <ImageTooltip {patterns} {labels} />
  {/await}
</div>
