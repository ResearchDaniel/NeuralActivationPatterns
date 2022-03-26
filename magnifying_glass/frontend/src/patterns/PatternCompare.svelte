<script lang="ts">
  import Pattern from "./Pattern.svelte";
  import MultiStatisticsChart from "./charts/MultiStatisticsChart.svelte";

  import {
    pinnedPatterns,
    filteredPinnedPatternUids,
    filteredPinnedPatterns,
  } from "../stores";

  let width: number;

  $: patternWidth = Math.max(
    width / $filteredPinnedPatternUids.length - 16,
    800
  );
</script>

<div class="flex flex-col w-full h-full min-h-0 p-2">
  <div class="w-full p-2 overflow-x-auto shrink-0">
    <MultiStatisticsChart patterns={$pinnedPatterns} />
  </div>
  <div class="flex min-w-0 overflow-x-auto pr-2" bind:clientWidth={width}>
    {#each $filteredPinnedPatternUids as uid, patternIndex}
      <Pattern
        {patternIndex}
        pattern={$pinnedPatterns[uid]}
        expanded={true}
        showStats={false}
        {patternWidth}
        filteredSamples={$filteredPinnedPatterns[uid].samples}
      />
    {/each}
  </div>
</div>
