<script lang="ts">
  import ImageTooltip from "./components/ImageTooltip.svelte";

  import Distribution from "./Distribution.svelte";
  import Patterns from "./patterns/Patterns.svelte";

  import type { PatternForSample } from "./types";

  export let model: string;
  export let layer: string;
  export let dataset: {
    file_name: string;
    label?: string;
    prediction?: string;
  }[];

  $: fetchPatterns = (async () => {
    const response = await fetch(`/api/get_patterns/${model}/${layer}`);
    const jsonResponse = await response.json();
    const patterns = JSON.parse(jsonResponse);
    return patterns.map((pattern, index) => {
      return {
        patternId: pattern.patternId,
        probability: pattern.probability,
        outlierScore: pattern.outlier_score,
        fileName: dataset[index].file_name,
        label: dataset[index].label,
        prediction: dataset[index].prediction,
      } as PatternForSample;
    });
  })();
</script>

<div class="flex flex-col flex-grow p-2 min-h-0">
  {#await fetchPatterns then patterns}
    <Distribution {patterns} />
    <Patterns {model} {layer} {patterns} />
    <ImageTooltip />
  {/await}
</div>
