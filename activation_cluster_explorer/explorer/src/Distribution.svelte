<script lang="ts">
  import SubHeading from "./components/SubHeading.svelte";

  import type { VisualizationSpec } from "vega-embed";
  import { VegaLite } from "svelte-vega";
  import type { EmbedOptions } from "vega-embed";

  import type { PatternForSample } from "./types";
  import { themeConfig } from "./constants";

  export let patterns: PatternForSample[];

  const spec: VisualizationSpec = {
    $schema: "https://vega.github.io/schema/vega-lite/v5.json",
    mark: { type: "circle", tooltip: true },
    data: { name: "table" },
    encoding: {
      x: { field: "patternId" },
      y: { field: "label" },
      size: { aggregate: "count" },
    },
  };
  const options = {
    actions: false,
    config: themeConfig,
  } as EmbedOptions;

  $: data = { table: patterns };
</script>

<div class="flex flex-col">
  <SubHeading heading={"Distribution"} />
  <VegaLite {data} {spec} {options} />
</div>
