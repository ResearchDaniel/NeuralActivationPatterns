<script lang="ts">
  import ProbabilityChart from "./ProbabilityChart.svelte";
  import LabelChart from "./LabelChart.svelte";
  import PredictionChart from "./PredictionChart.svelte";
  import StatisticsChart from "./StatisticsChart.svelte";

  import {
    showLabels,
    showPredictions,
    showProbability,
    showStatistics,
  } from "../../stores";

  import type { PatternForSample, Pattern } from "../../types";

  export let pattern: Pattern;
  export let filteredSamples: PatternForSample[];

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
