<script lang="ts">
  import PatternImage from "./images/PatternImage.svelte";
  import PatternImageList from "./images/PatternImageList.svelte";
  import Charts from "./charts/Charts.svelte";

  import type { PatternForSample, Pattern } from "../types";

  import {
    numCenters,
    numOutliers,
    showAverage,
    showLabels,
    showPredictions,
    showProbability,
    showStatistics,
  } from "../stores";

  export let pattern: Pattern;
  export let filteredSamples: PatternForSample[];
  export let patternId: number;
  export let model: string;
  export let layer: string;
  export let expanded: boolean;

  $: centers = filteredSamples.slice(0, $numCenters);
  $: derivedNumOutliers =
    filteredSamples.length >= $numCenters + $numOutliers
      ? $numOutliers
      : filteredSamples.length - $numCenters;
  $: outliers =
    derivedNumOutliers > 0 ? filteredSamples.slice(-derivedNumOutliers) : [];
</script>

<div class="flex flex-wrap" class:overflow-y-auto={expanded}>
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
  {#if $showProbability || $showLabels || $showPredictions || (pattern.statistics !== undefined && $showStatistics && !expanded)}
    <Charts {pattern} {filteredSamples} {expanded} />
  {/if}
</div>
