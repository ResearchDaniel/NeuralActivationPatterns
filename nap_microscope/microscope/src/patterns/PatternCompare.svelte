<script lang="ts">
  import Pattern from "./Pattern.svelte";

  import {
    pinnedPatterns,
    filteredPinnedPatternUids,
    filteredPinnedPatterns,
  } from "../stores";
  import MultiStatisticsChart from "./charts/MultiStatisticsChart.svelte";

  let width: number;

  $: patternWidth = Math.max(
    width / $filteredPinnedPatternUids.length - 16,
    800
  );
</script>

<div class="flex flex-col w-full h-full">
  <div class="w-full p-2 overflow-x-auto shrink-0">
    <MultiStatisticsChart />
  </div>
  <div class="flex min-w-0 overflow-x-auto h-full p-2" bind:clientWidth={width}>
    {#each $filteredPinnedPatternUids as uid, patternIndex}
      <Pattern
        {patternIndex}
        pattern={$pinnedPatterns[uid]}
        expanded={true}
        {patternWidth}
        filteredSamples={$filteredPinnedPatterns[uid].samples}
      />
    {/each}
  </div>
</div>
