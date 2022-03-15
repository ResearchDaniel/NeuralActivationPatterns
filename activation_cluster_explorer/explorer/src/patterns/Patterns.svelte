<script lang="ts">
  import Pattern from "./Pattern.svelte";
  import SubHeading from "../elements/SubHeading.svelte";

  import type { PatternForSample } from "../types";
  import { imageFilter, labelFilter, predictionFilter } from "../stores";
  import { filterPatterns } from "../helpers";

  export let patterns: PatternForSample[];
  export let persistence: number[];

  $: filteredPatterns = filterPatterns(
    patterns,
    $labelFilter,
    $predictionFilter,
    $imageFilter
  );
  $: patternIds = [
    ...new Set(filteredPatterns.map((element) => element.patternId)),
  ].sort((a, b) => persistence[b] - persistence[a]);
</script>

<div class="flex flex-col min-h-0">
  <SubHeading heading={`Patterns (${patternIds.length})`} />
  <div class="flex flex-col items-start overflow-y-auto min-h-0 pt-2">
    {#each patternIds as patternId}
      <Pattern
        samples={patterns.filter((sample) => sample.patternId === patternId)}
        filteredSamples={filteredPatterns.filter(
          (sample) => sample.patternId === patternId
        )}
      />
    {/each}
  </div>
</div>
