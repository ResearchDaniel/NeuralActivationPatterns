<script lang="ts">
  import Pattern from "./Pattern.svelte";
  import SubHeading from "../components/SubHeading.svelte";

  import type { PatternForSample } from "../types";

  export let patterns: PatternForSample[];

  export let model: string;
  export let layer: string;

  $: patternIds = [
    ...new Set(patterns.map((element) => element.patternId)),
  ].sort();
</script>

<div class="flex flex-col min-h-0">
  <SubHeading heading={"Clusters"} />
  <div class="flex flex-col items-start overflow-y-auto min-h-0">
    {#each patternIds as patternId}
      <Pattern
        samples={patterns.filter((sample) => sample.patternId === patternId)}
        {patternId}
        {model}
        {layer}
      />
    {/each}
  </div>
</div>
