<script lang="ts">
  import PatternsList from "./patterns/Patterns.svelte";
  import { patternFilter } from "./stores";

  import type { PatternForSample, Patterns } from "./types";

  export let patterns: Patterns;

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

<div class="flex flex-col pl-2 pr-2 min-h-0">
  <PatternsList patterns={filteredSamples} persistence={patterns.persistence} />
</div>
