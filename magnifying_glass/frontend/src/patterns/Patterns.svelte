<script lang="ts">
  import { startswith } from "arquero";

  import Pattern from "./Pattern.svelte";
  import SubHeading from "../elements/SubHeading.svelte";

  import type { PatternForSample, Patterns } from "../types";
  import { compactPatterns, labelFilter, predictionFilter } from "../stores";
  import { filterPatterns } from "../helpers";

  export let patterns: Patterns;
  export let filteredSamples: PatternForSample[];
  export let persistence: number[];

  $: filteredPatterns = filterPatterns(
    filteredSamples,
    $labelFilter,
    $predictionFilter
  );
  $: patternIds = [
    ...new Set(filteredPatterns.map((element) => element.patternId)),
  ].sort((a, b) => persistence[b] - persistence[a]);
</script>

<div class="flex flex-col min-h-0">
  <SubHeading heading={`Patterns (${patternIds.length})`} />
  <div
    class="flex items-start overflow-y-auto min-h-0 pt-2"
    class:flex-wrap={$compactPatterns}
    class:flex-col={!$compactPatterns}
  >
    {#each patternIds as patternId}
      <Pattern
        patternIndex={patternId}
        pattern={{
          samples: patterns.samples.filter(
            (sample) => sample.patternId === patternId
          ),
          persistence: patterns.persistence[patternId],
        }}
        filteredSamples={filteredPatterns.filter(
          (sample) => sample.patternId === patternId
        )}
      />
    {/each}
  </div>
</div>
