<script lang="ts">
  import type { VegaLiteSpec } from "svelte-vega";
  import type { EmbedOptions } from "vega-embed";
  import { VegaLite } from "svelte-vega";

  import { tableau20, themeConfig } from "../../constants";
  import { filteredPinnedPatternUids } from "../../stores";

  import type ColumnTable from "arquero/dist/types/table/column-table";

  export let statsTable: ColumnTable;

  const options = {
    config: themeConfig,
    actions: false,
  } as EmbedOptions;

  $: spec = {
    $schema: "https://vega.github.io/schema/vega-lite/v5.json",
    data: {
      values: statsTable,
    },
    height: 100,
    encoding: { x: { field: "unit", type: "nominal", title: "unit" } },
    layer: [
      {
        mark: { type: "circle" },
        encoding: {
          y: { field: "mean", type: "quantitative", title: "activation" },
          xOffset: { field: "key" },
          color: { field: "key" },
        },
      },
      {
        mark: { type: "bar", opacity: 0.3 },
        encoding: {
          y: { field: "q1", type: "quantitative" },
          y2: { field: "q3" },
          xOffset: { field: "key" },
          color: { field: "key" },
        },
      },
      {
        mark: { type: "rule", opacity: 0.3 },
        encoding: {
          y: {
            field: "lower",
            type: "quantitative",
          },
          xOffset: { field: "key" },
          color: {
            field: "key",
            scale: {
              domain: $filteredPinnedPatternUids,
              range: tableau20,
            },
            legend: null,
          },
          y2: { field: "upper" },
        },
      },
      {
        mark: "rule",
        encoding: {
          opacity: { value: 0 },
          tooltip: [
            { field: "mean", type: "quantitative" },
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
