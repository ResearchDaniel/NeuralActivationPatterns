<script lang="ts">
  import { table } from "arquero";

  import { removeZerosStatistics } from "../stores";
  import { fetchPatternStatistics } from "../api";

  import Pattern from "./Pattern.svelte";
  import MultiStatisticsChart from "./charts/MultiStatisticsChart.svelte";

  import {
    pinnedPatterns,
    filteredPinnedPatternUids,
    filteredPinnedPatterns,
    showStatistics,
  } from "../stores";

  let width: number;

  $: fetchStatsTable = Promise.all(
    Object.keys($pinnedPatterns).map((key) =>
      fetchPatternStatistics(
        $pinnedPatterns[key].samples[0].model,
        $pinnedPatterns[key].samples[0].layer,
        $pinnedPatterns[key].samples[0].patternId,
        key
      )
    )
  ).then((statistics) => {
    const statsTables = statistics
      .map((stats) => {
        if (stats !== undefined) {
          let statsTable = table({
            key: Array(stats.table.numRows()).fill(stats.key),
          });
          statsTable = stats.table.assign(statsTable);
          if ($removeZerosStatistics) {
            statsTable = statsTable.filter(
              (element) => element.upper > 0.05 || element.lower < -0.05
            );
          }
          return statsTable;
        }
      })
      .filter((table) => table !== undefined);
    if (statsTables.length === 0) return undefined;
    else if (statsTables.length === 1) return statsTables[0];
    return statsTables.shift().concat(statsTables);
  });
  $: patternWidth = Math.max(
    width / $filteredPinnedPatternUids.length - 16,
    800
  );
</script>

<div class="flex flex-col w-full h-full min-h-0 p-2">
  {#if $showStatistics}
    {#await fetchStatsTable then statsTable}
      {#if statsTable !== undefined}
        <div class="w-full p-2 overflow-x-auto shrink-0">
          <MultiStatisticsChart {statsTable} />
        </div>
      {/if}
    {/await}
  {/if}
  <div class="flex min-w-0 overflow-x-auto pr-2 grow" bind:clientWidth={width}>
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
