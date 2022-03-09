<script lang="ts">
  import Pattern from "./Pattern.svelte";
  import SubHeading from "../components/SubHeading.svelte";

  import type { PatternForSample } from "../types";
  import { labelFilter, predictionFilter } from "../stores";

  export let patterns: PatternForSample[];
  export let persistence: number[];

  $: filteredPatterns = patterns.filter((pattern) => {
    if ($labelFilter.length !== 0) {
      if (!$labelFilter.includes(`${pattern.label}`)) {
        return false;
      }
    }
    if ($predictionFilter.length !== 0) {
      if (!$predictionFilter.includes(`${pattern.prediction}`)) {
        return false;
      }
    }
    return true;
  });
  $: patternIds = [
    ...new Set(filteredPatterns.map((element) => element.patternId)),
  ].sort((a, b) => persistence[b] - persistence[a]);
</script>

<div class="flex flex-col min-h-0">
  <SubHeading heading={`Patterns (${patternIds.length})`} />
  <div class="flex flex-col items-start overflow-y-auto min-h-0">
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
