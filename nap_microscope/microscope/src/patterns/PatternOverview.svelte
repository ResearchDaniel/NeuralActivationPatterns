<script lang="ts">
  import PatternImage from "./PatternImage.svelte";
  import PatternImageList from "./PatternImageList.svelte";
  import ProbabilityChart from "./ProbabilityChart.svelte";
  import LabelChart from "./LabelChart.svelte";
  import PredictionChart from "./PredictionChart.svelte";
  import StatisticsChart from "./StatisticsChart.svelte";

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
  $: metadata = pattern.samples.reduce(sampleMetadata, {
    labels: {},
    predictions: {},
  });
  $: filteredMetadata = filteredSamples.reduce(sampleMetadata, {
    labels: {},
    predictions: {},
  });

  function sampleMetadata(
    aggregate: {
      labels: Record<string, number>;
      predictions: Record<string, number>;
    },
    sample: PatternForSample
  ) {
    if (sample.label in aggregate.labels) {
      aggregate.labels[sample.label]++;
    } else {
      aggregate.labels[sample.label] = 1;
    }
    if (sample.prediction in aggregate.predictions) {
      aggregate.predictions[sample.prediction]++;
    } else {
      aggregate.predictions[sample.prediction] = 1;
    }
    return aggregate;
  }
</script>

<div class="flex flex-wrap" class:overflow-y-auto={expanded}>
  {#if $showAverage}
    <div class="flex flex-col pr-4">
      <p>Average</p>
      <PatternImage
        imagePath={pattern.samples[0].filter !== undefined
          ? `/api/get_filter_average/${model}/${layer}/${pattern.samples[0].filterMethod}/${pattern.samples[0].filter}/${patternId}`
          : `/api/get_average/${model}/${layer}/${patternId}`}
      />
    </div>
  {/if}
  {#if !expanded}
    <div class="flex flex-col pr-4">
      <p>Most Salient</p>
      <PatternImageList {model} samples={centers} {layer} />
    </div>
    {#if outliers.length > 0}
      <div class="flex flex-col pr-4">
        <p>Least Salient</p>
        <PatternImageList {model} samples={outliers} {layer} />
      </div>
    {/if}
  {/if}
  {#if $showProbability || $showLabels || $showPredictions || (pattern.statistics !== undefined && $showStatistics)}
    <div class="flex flex-col min-w-0">
      <p>Distribution</p>
      <div class="flex flex-wrap">
        {#if pattern.statistics !== undefined && $showStatistics}
          <div class="min-w-0 overflow-x-auto">
            <StatisticsChart statistics={pattern.statistics} />
          </div>
        {/if}
        {#if $showProbability}
          <div class="min-w-0 overflow-x-auto">
            <ProbabilityChart samples={pattern.samples} />
          </div>
        {/if}
        {#if $showLabels}
          <div class="min-w-0 overflow-x-auto">
            <LabelChart {metadata} {filteredMetadata} />
          </div>
        {/if}
        {#if $showPredictions}
          <div class="min-w-0 overflow-x-auto">
            <PredictionChart {metadata} {filteredMetadata} />
          </div>
        {/if}
      </div>
    </div>
  {/if}
</div>
