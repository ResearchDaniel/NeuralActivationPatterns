<script lang="ts">
  import Pattern from "./Pattern.svelte";
  import SubHeading from "../components/SubHeading.svelte";

  import type { PatternForSample } from "../types";

  export let patterns: PatternForSample[];

  $: patternIds = [
    ...new Set(patterns.map((element) => element.patternId)),
  ].sort();
</script>

<div class="flex flex-col min-h-0">
  <SubHeading heading={`Clusters (${patternIds.length})`} />
  <div class="flex flex-col items-start overflow-y-auto min-h-0">
    {#each patternIds as patternId}
      <Pattern
        samples={patterns.filter((sample) => sample.patternId === patternId)}
      />
    {/each}
  </div>
</div>
