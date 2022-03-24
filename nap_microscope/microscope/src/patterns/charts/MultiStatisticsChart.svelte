<script lang="ts">
  import type { VegaLiteSpec } from "svelte-vega";
  import type { EmbedOptions } from "vega-embed";
  import { VegaLite } from "svelte-vega";

  import { pinnedPatterns } from "../../stores";
  import { themeConfig } from "../../constants";

  const options = {
    config: themeConfig,
    actions: false,
  } as EmbedOptions;

  $: stats = Object.keys($pinnedPatterns)
    .map((key) => {
      const keyStats = $pinnedPatterns[key].statistics;
      const mappedKeyStats = keyStats.min.map((min, index) => {
        return {
          index: index,
          lower: min,
          upper: keyStats.max[index],
          q1: keyStats.q1[index],
          q3: keyStats.q3[index],
          median: keyStats.mean[index],
          key: key,
        };
      });
      return mappedKeyStats;
    })
    .flat();
  $: spec = {
    $schema: "https://vega.github.io/schema/vega-lite/v5.json",
    data: {
      values: stats,
    },
    height: 100,
    encoding: { x: { field: "index", type: "nominal", title: "unit" } },
    layer: [
      {
        mark: { type: "circle" },
        encoding: {
          y: { field: "median", type: "quantitative", title: "activation" },
          xOffset: { field: "key" },
          color: { field: "key" },
        },
      },
      {
        mark: { type: "bar", color: "#0071e355" },
        encoding: {
          y: { field: "q1", type: "quantitative" },
          y2: { field: "q3" },
          xOffset: { field: "key" },
          color: { field: "key" },
        },
      },
      {
        mark: { type: "rule", color: "#0071e355" },
        encoding: {
          y: {
            field: "lower",
            type: "quantitative",
          },
          xOffset: { field: "key" },
          color: {
            field: "key",
            scale: { scheme: "tableau20" },
            legend: null,
          },
          y2: { field: "upper" },
        },
      },
    ],
  } as VegaLiteSpec;
</script>

<VegaLite {spec} {options} />
