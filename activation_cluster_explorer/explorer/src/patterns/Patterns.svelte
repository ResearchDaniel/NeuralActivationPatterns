<script lang="ts">
  import Pattern from "./Pattern.svelte";
  import SubHeading from "../components/SubHeading.svelte";

  import type { PatternForSample } from "../types";

  export let patterns: PatternForSample[];
  export let model: string;
  export let layer: string;

  let selectedPattern: number | undefined;

  $: patternIds = [
    ...new Set(patterns.map((element) => element.patternId)),
  ].sort();
</script>

<div class="flex flex-col min-h-0">
  {#if selectedPattern === undefined}
    <SubHeading heading={`Clusters (${patternIds.length})`} />
    <div class="flex flex-col items-start overflow-y-auto min-h-0">
      {#each patternIds as patternId}
        <Pattern
          samples={patterns.filter((sample) => sample.patternId === patternId)}
          {patternId}
          {model}
          {layer}
          on:zoom={() => (selectedPattern = patternId)}
        />
      {/each}
    </div>
  {:else}
    <Pattern
      samples={patterns.filter(
        (sample) => sample.patternId === selectedPattern
      )}
      patternId={selectedPattern}
      {model}
      {layer}
      on:zoom={() => (selectedPattern = undefined)}
      expanded={true}
    />
  {/if}
</div>
