<script lang="ts">
  import type { VegaLiteSpec } from "svelte-vega";
  import type { EmbedOptions } from "vega-embed";
  import { VegaLite } from "svelte-vega";

  import { themeConfig } from "../constants";
  import { predictionFilter } from "../stores";

  export let metadata: {
    labels: Record<string, number>;
    predictions: Record<string, number>;
  };
  export let filteredMetadata: {
    labels: Record<string, number>;
    predictions: Record<string, number>;
  };

  const options = {
    config: themeConfig,
    actions: false,
  } as EmbedOptions;

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
</script>

<VegaLite
  spec={layeredPredictionSpec}
  {options}
  signalListeners={{ select: handleSelectionPrediction }}
/>
