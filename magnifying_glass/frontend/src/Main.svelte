<script lang="ts">
  import Explainability from "./explainability/Explainability.svelte";
  import PatternsList from "./patterns/Patterns.svelte";
  import { patternFilter } from "./stores";

  import type { PatternForSample, Patterns } from "./types";

  export let patterns: Patterns;
  export let maxActivating: string[];

  $: sortedSamples = patterns.samples.sort(
    (a: PatternForSample, b: PatternForSample) => b.probability - a.probability
  );
  $: filteredSamples =
    $patternFilter.length === 0
      ? sortedSamples
      : sortedSamples.filter((pattern) => {
          for (let filter of $patternFilter) {
            if (
              pattern.label === filter.label &&
              pattern.patternId === filter.patternId
            ) {
              return true;
            }
          }
          return false;
        });
</script>

<div class="flex flex-row pl-2 pr-2 min-h-0">
  <PatternsList
    {patterns}
    {filteredSamples}
    persistence={patterns.persistence}
  />
  {#if maxActivating.length > 0}
    <Explainability {maxActivating} />
  {/if}
</div>
