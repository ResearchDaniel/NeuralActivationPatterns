<script lang="ts">
  import PatternImage from "./PatternImage.svelte";
  import PatternImageList from "./PatternImageList.svelte";
  import type { PatternForSample } from "../types";

  import {
    numCenters,
    numOutliers,
    showAverage,
    showDistribution,
    showLabels,
    showPredictions,
    showProbability,
  } from "../stores";
  import ProbabilityChart from "./ProbabilityChart.svelte";
  import LabelChart from "./LabelChart.svelte";
  import PredictionChart from "./PredictionChart.svelte";

  export let samples: PatternForSample[];
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
  $: metadata = samples.reduce(sampleMetadata, {
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
        imagePath={samples[0].filter !== undefined
          ? `/api/get_filter_average/${model}/${layer}/${samples[0].filterMethod}/${samples[0].filter}/${patternId}`
          : `/api/get_average/${model}/${layer}/${patternId}`}
      />
    </div>
  {/if}
  {#if !expanded}
    <div class="flex flex-col pr-4">
      <p>Centers</p>
      <PatternImageList {model} samples={centers} {layer} />
    </div>
    {#if outliers.length > 0}
      <div class="flex flex-col pr-4">
        <p>Outliers</p>
        <PatternImageList {model} samples={outliers} {layer} />
      </div>
    {/if}
  {/if}
  {#if $showProbability || $showLabels || $showPredictions}
    <div class="flex flex-col min-w-0">
      <p>Distribution</p>
      <div class="flex flex-wrap">
        {#if $showProbability}
          <div class="min-w-0 overflow-x-auto">
            <ProbabilityChart {samples} />
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
