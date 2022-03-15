<script lang="ts">
  import type { VegaLiteSpec } from "svelte-vega";
  import type { EmbedOptions } from "vega-embed";
  import { VegaLite } from "svelte-vega";

  import PatternImage from "./PatternImage.svelte";
  import PatternImageList from "./PatternImageList.svelte";
  import type { PatternForSample } from "../types";

  import {
    labelFilter,
    numCenters,
    numOutliers,
    predictionFilter,
    showAverage,
  } from "../stores";
  import { themeConfig } from "../constants";

  export let samples: PatternForSample[];
  export let filteredSamples: PatternForSample[];
  export let patternId: number;
  export let model: string;
  export let layer: string;
  export let expanded: boolean;

  const options = {
    config: themeConfig,
    actions: false,
  } as EmbedOptions;

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
  $: extent = [
    samples[samples.length - 1].probability === 1
      ? 0
      : samples[samples.length - 1].probability,
    1,
  ];
  $: probabilityHistogramSpec = {
    $schema: "https://vega.github.io/schema/vega-lite/v5.json",
    data: { values: samples },
    width: 100,
    height: 100,
    mark: { type: "bar", tooltip: true },
    encoding: {
      x: {
        bin: { extent: extent },
        field: "probability",
      },
      y: { aggregate: "count", title: "samples" },
    },
  } as VegaLiteSpec;
  $: labelSpec = {
    $schema: "https://vega.github.io/schema/vega-lite/v5.json",
    description: "A simple bar chart with embedded data.",
    data: {
      values: Object.keys(metadata.labels).map((key) => {
        return {
          label: key,
          samples: metadata.labels[key],
        };
      }),
    },
    params: [{ name: "select", select: { type: "point", encodings: ["x"] } }],
    height: 100,
    mark: { type: "bar", tooltip: true },
    encoding: {
      x: { field: "label", type: "nominal" },
      y: { field: "samples", type: "quantitative" },
    },
  } as VegaLiteSpec;
  $: layeredLabelSpec = {
    layer: [
      { ...labelSpec, mark: { ...labelSpec.mark, color: "#dcdcdc" } },
      {
        ...labelSpec,
        data: {
          values: Object.keys(filteredMetadata.labels).map((key) => {
            return {
              label: key,
              samples: filteredMetadata.labels[key],
            };
          }),
        },
        params: [],
      },
    ],
  } as VegaLiteSpec;
  $: predictionSpec = {
    $schema: "https://vega.github.io/schema/vega-lite/v5.json",
    description: "A simple bar chart with embedded data.",
    data: {
      values: Object.keys(metadata.predictions).map((key) => {
        return {
          prediction: key,
          samples: metadata.predictions[key],
        };
      }),
    },
    params: [{ name: "select", select: { type: "point", encodings: ["x"] } }],
    height: 100,
    mark: { type: "bar", tooltip: true },
    encoding: {
      x: { field: "prediction", type: "nominal" },
      y: { field: "samples", type: "quantitative" },
    },
  } as VegaLiteSpec;
  $: layeredPredictionSpec = {
    layer: [
      {
        ...predictionSpec,
        mark: { ...predictionSpec.mark, color: "#dcdcdc" },
      },
      {
        ...predictionSpec,
        data: {
          values: Object.keys(filteredMetadata.predictions).map((key) => {
            return {
              prediction: key,
              samples: filteredMetadata.predictions[key],
            };
          }),
        },
        params: [],
      },
    ],
  } as VegaLiteSpec;

  function handleSelectionLabel(...args: any) {
    if (args[1].label !== undefined) {
      const index = $labelFilter.indexOf(args[1].label[0], 0);
      if (index > -1) {
        labelFilter.update((filters) => {
          filters.splice(index, 1);
          return filters;
        });
      } else {
        labelFilter.update((filters) => [
          ...new Set([...filters, ...args[1].label]),
        ]);
      }
    }
  }

  function handleSelectionPrediction(...args: any) {
    if (args[1].prediction !== undefined) {
      const index = $predictionFilter.indexOf(args[1].prediction[0], 0);
      if (index > -1) {
        predictionFilter.update((filters) => {
          filters.splice(index, 1);
          return filters;
        });
      } else {
        predictionFilter.update((filters) => [
          ...new Set([...filters, ...args[1].prediction]),
        ]);
      }
    }
  }

  function sampleMetadata(aggregate, sample) {
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
  <div class="flex flex-col min-w-0">
    <p>Distribution</p>
    <div class="flex flex-wrap">
      <div class="min-w-0 overflow-x-auto">
        <VegaLite spec={probabilityHistogramSpec} {options} />
      </div>
      <div class="min-w-0 overflow-x-auto">
        <VegaLite
          spec={layeredLabelSpec}
          {options}
          signalListeners={{ select: handleSelectionLabel }}
        />
      </div>
      <div class="min-w-0 overflow-x-auto">
        <VegaLite
          spec={layeredPredictionSpec}
          {options}
          signalListeners={{ select: handleSelectionPrediction }}
        />
      </div>
    </div>
  </div>
</div>
