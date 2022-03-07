<script lang="ts">
  import PatternImage from "./PatternImage.svelte";
  import PatternImageList from "./PatternImageList.svelte";
  import type { PatternForSample } from "../types";

  import { numCenters, numOutliers } from "../stores";
  import { themeConfig } from "../constants";

  import type { VegaLiteSpec } from "svelte-vega";
  import type { EmbedOptions } from "vega-embed";
  import { VegaLite } from "svelte-vega";

  export let samples: PatternForSample[];
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
  $: metadata = samples.reduce(
    function (aggregate, sample) {
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
    },
    { labels: {}, predictions: {} }
  );
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
    height: 100,
    mark: { type: "bar", tooltip: true },
    encoding: {
      x: { field: "label", type: "nominal" },
      y: { field: "samples", type: "quantitative" },
    },
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
    height: 100,
    mark: { type: "bar", tooltip: true },
    encoding: {
      x: { field: "prediction", type: "nominal" },
      y: { field: "samples", type: "quantitative" },
    },
  } as VegaLiteSpec;
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
      <VegaLite spec={labelSpec} {options} />
      <VegaLite spec={predictionSpec} {options} />
    </div>
  </div>
</div>
