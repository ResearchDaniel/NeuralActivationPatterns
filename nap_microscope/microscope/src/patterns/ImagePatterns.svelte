<script lang="ts">
  import PatternComponent from "./Pattern.svelte";

  import { filterPattern } from "../helpers";

  import type { Pattern } from "../types";

  import { labelFilter, predictionFilter } from "../stores";

  export let patterns: Pattern[];
  export let width: number;

  $: patternsWithFilteredSamples = patterns
    .map((pattern) => {
      return {
        pattern: pattern,
        filteredSamples: filterPattern(pattern, $labelFilter, $predictionFilter)
          .samples,
      };
    })
    .filter((item) => item.filteredSamples.length > 0);
  $: patternWidth = Math.max(
    width / patternsWithFilteredSamples.length - 20,
    800
  );
</script>

{#each patternsWithFilteredSamples as item}
  <PatternComponent
    pattern={item.pattern}
    expanded={true}
    {patternWidth}
    filteredSamples={item.filteredSamples}
  />
{/each}
