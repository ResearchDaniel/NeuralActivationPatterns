<script lang="ts">
  import Patterns from "./patterns/Patterns.svelte";
  import { patternFilter } from "./stores";

  import type { PatternForSample } from "./types";

  export let patterns: PatternForSample[];

  $: sortedPatterns = patterns.sort(
    (a: PatternForSample, b: PatternForSample) => b.probability - a.probability
  );
  $: filteredPatterns =
    $patternFilter.length === 0
      ? sortedPatterns
      : sortedPatterns.filter((pattern) => {
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

<div class="flex flex-col flex-grow p-2 min-h-0">
  <Patterns patterns={filteredPatterns} />
</div>
