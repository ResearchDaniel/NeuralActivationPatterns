<script lang="ts">
  import type { VegaLiteSpec } from "svelte-vega";
  import type { EmbedOptions } from "vega-embed";
  import { VegaLite } from "svelte-vega";

  import type { Statistics } from "../../types";
  import { themeConfig } from "../../constants";
  import { removeZerosStatistics } from "../../stores";

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
  $: filteredStatistics = $removeZerosStatistics
    ? mappedStatistics.filter(
        (element) => element.upper > 0.05 || element.lower < -0.05
      )
    : mappedStatistics;
  $: spec = {
    $schema: "https://vega.github.io/schema/vega-lite/v5.json",
    data: {
      values: filteredStatistics,
    },
    height: 100,
    encoding: { x: { field: "index", type: "nominal", title: "unit" } },
    layer: [
      {
        mark: { type: "circle" },
        encoding: {
          y: { field: "median", type: "quantitative", title: "activation" },
        },
      },
      {
        mark: { type: "bar", opacity: 0.3 },
        encoding: {
          y: { field: "q1", type: "quantitative" },
          y2: { field: "q3" },
        },
      },
      {
        mark: { type: "rule", opacity: 0.3 },
        encoding: {
          y: {
            field: "lower",
            type: "quantitative",
          },
          y2: { field: "upper" },
        },
      },
      {
        mark: "rule",
        encoding: {
          opacity: { value: 0 },
          tooltip: [
            { field: "median", type: "quantitative" },
            { field: "lower", type: "quantitative" },
            { field: "upper", type: "quantitative" },
            { field: "q1", type: "quantitative" },
            { field: "q3", type: "quantitative" },
          ],
        },
        params: [
          {
            name: "hover",
            select: {
              type: "point",
              fields: ["index"],
              nearest: true,
              on: "mouseover",
              clear: "mouseout",
            },
          },
        ],
      },
    ],
  } as VegaLiteSpec;
</script>

<VegaLite {spec} {options} />
