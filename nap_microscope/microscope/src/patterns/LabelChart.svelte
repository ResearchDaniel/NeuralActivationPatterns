<script lang="ts">
  import type { VegaLiteSpec } from "svelte-vega";
  import type { EmbedOptions } from "vega-embed";
  import { VegaLite } from "svelte-vega";

  import { themeConfig } from "../constants";
  import { labelFilter } from "../stores";

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
</script>

<VegaLite
  spec={layeredLabelSpec}
  {options}
  signalListeners={{ select: handleSelectionLabel }}
/>
