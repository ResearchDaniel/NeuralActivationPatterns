<script lang="ts">
  import type { VegaLiteSpec } from "svelte-vega";
  import type { EmbedOptions } from "vega-embed";
  import { VegaLite } from "svelte-vega";

  import type { Statistics } from "../types";
  import { themeConfig } from "../constants";

  export let statistics: Statistics;

  const options = {
    config: themeConfig,
    actions: false,
  } as EmbedOptions;

  $: mappedStatistics = statistics.min.map((min, index) => {
    return {
      index: index,
      lower: min,
      upper: statistics.max[index],
      q1: statistics.q1[index],
      q3: statistics.q3[index],
      median: statistics.mean[index],
    };
  });
  $: spec = {
    $schema: "https://vega.github.io/schema/vega-lite/v5.json",
    data: {
      values: mappedStatistics,
    },
    height: 100,
    encoding: { x: { field: "index", type: "nominal", title: "unit" } },
    layer: [
      {
        mark: { type: "rule" },
        encoding: {
          y: {
            field: "lower",
            type: "quantitative",
            scale: { zero: false },
            title: "activation",
          },
          y2: { field: "upper" },
        },
      },
      {
        mark: { type: "bar", color: "black" },
        encoding: {
          y: { field: "q1", type: "quantitative" },
          y2: { field: "q3" },
          color: { field: "Species", type: "nominal", legend: null },
        },
      },
      {
        mark: { type: "circle" },
        encoding: {
          y: { field: "median", type: "quantitative" },
        },
      },
    ],
  } as VegaLiteSpec;
</script>

<VegaLite {spec} {options} />
