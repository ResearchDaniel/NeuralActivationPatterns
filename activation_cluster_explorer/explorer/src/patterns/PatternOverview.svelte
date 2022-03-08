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
  } from "../stores";
  import { themeConfig } from "../constants";

  export let samples: PatternForSample[];
  export let filteredSamples: PatternForSample[];
  export let patternId: number;
  export let model: string;
  export let layer: string;

  const probabilityHistogramSpec: VegaLiteSpec = {
    $schema: "https://vega.github.io/schema/vega-lite/v5.json",
    data: { name: "table" },
    width: 100,
    height: 100,
    mark: { type: "bar", tooltip: true },
    encoding: {
      x: { bin: true, field: "probability" },
      y: { aggregate: "count", title: "samples" },
    },
  };
  const options = {
    config: themeConfig,
    actions: false,
  } as EmbedOptions;

  $: centers = samples.slice(0, $numCenters);
  $: outliers = samples.slice(-$numOutliers);
  $: data = { table: samples };
  $: metadata = samples.reduce(sampleMetadata, { labels: {}, predictions: {} });
  $: filteredMetadata = filteredSamples.reduce(sampleMetadata, {
    labels: {},
    predictions: {},
  });
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
  };
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
      { ...predictionSpec, mark: { ...predictionSpec.mark, color: "#dcdcdc" } },
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
  };

  function handleSelectionLabel(...args: any) {
    if (args[1].label !== undefined) {
      labelFilter.update((filters) => [
        ...new Set([...filters, ...args[1].label]),
      ]);
    }
  }

  function handleSelectionPrediction(...args: any) {
    if (args[1].prediction !== undefined) {
      predictionFilter.update((filters) => [
        ...new Set([...filters, ...args[1].prediction]),
      ]);
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

<div class="flex flex-wrap">
  <div class="flex flex-col pr-4">
    <p>Average</p>
    <PatternImage
      imagePath={`/api/get_average/${model}/${layer}/${patternId}`}
    />
  </div>
  <div class="flex flex-col pr-4">
    <p>Centers</p>
    <PatternImageList {model} samples={centers} {layer} />
  </div>
  <div class="flex flex-col pr-4">
    <p>Outliers</p>
    <PatternImageList {model} samples={outliers} {layer} />
  </div>
  <div class="flex flex-col">
    <p>Distribution</p>
    <div class="flex flex-wrap">
      <VegaLite {data} spec={probabilityHistogramSpec} {options} />
      <VegaLite
        spec={layeredLabelSpec}
        {options}
        signalListeners={{ select: handleSelectionLabel }}
      />
      <VegaLite
        spec={layeredPredictionSpec}
        {options}
        signalListeners={{ select: handleSelectionPrediction }}
      />
    </div>
  </div>
</div>
