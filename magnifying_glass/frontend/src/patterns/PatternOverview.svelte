<script lang="ts">
  import { table } from "arquero";

  import PatternImage from "./images/PatternImage.svelte";
  import PatternImageList from "./images/PatternImageList.svelte";
  import Charts from "./charts/Charts.svelte";

  import type { PatternForSample, Pattern } from "../types";

  import {
    numCenters,
    numOutliers,
    removeZerosStatistics,
    showAverage,
    showLabels,
    showOverviewStatistics,
    showPredictions,
    showProbability,
  } from "../stores";
  import { fetchPatternStatistics } from "../api";

  export let pattern: Pattern;
  export let filteredSamples: PatternForSample[];
  export let patternId: number;
  export let model: string;
  export let layer: string;
  export let expanded: boolean;

  $: fetchStatsTable = fetchPatternStatistics(
    model,
    layer,
    patternId,
    pattern.samples[0].patternUid
  ).then((statistics) => {
    if (statistics !== undefined) {
      let statsTable = table({
        key: Array(statistics.table.numRows()).fill(statistics.key),
      });
      statsTable = statistics.table.assign(statsTable);
      if ($removeZerosStatistics) {
        statsTable = statsTable.filter(
          (element) => element.upper > 0.05 || element.lower < -0.05
        );
      }
      return statsTable;
    }
  });
  $: centers = filteredSamples.slice(0, $numCenters);
  $: derivedNumOutliers =
    filteredSamples.length >= $numCenters + $numOutliers
      ? $numOutliers
      : filteredSamples.length - $numCenters;
  $: outliers =
    derivedNumOutliers > 0 ? filteredSamples.slice(-derivedNumOutliers) : [];
</script>

<div class="flex flex-wrap shrink-0" class:overflow-y-auto={expanded}>
  {#if $showAverage}
    <div class="flex flex-col pr-4">
      <p>Average</p>
      <PatternImage
        imagePath={`/api/get_average/${model}/${layer}/${patternId}`}
      />
    </div>
  {/if}
  {#if !expanded}
    <div class="flex flex-col pr-4">
      <p>Most Stable</p>
      <PatternImageList {model} samples={centers} {layer} />
    </div>
    {#if outliers.length > 0}
      <div class="flex flex-col pr-4">
        <p>Least Stable</p>
        <PatternImageList {model} samples={outliers} {layer} />
      </div>
    {/if}
  {/if}
  {#if $showOverviewStatistics}
    {#await fetchStatsTable then statsTable}
      {#if statsTable !== undefined}
        <Charts {pattern} {filteredSamples} {expanded} {statsTable} />
      {:else if $showProbability || $showLabels || $showPredictions}
        <Charts {pattern} {filteredSamples} {expanded} />
      {/if}
    {/await}
  {:else if $showProbability || $showLabels || $showPredictions}
    <Charts {pattern} {filteredSamples} {expanded} />
  {/if}
</div>
