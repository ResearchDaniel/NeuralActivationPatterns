<script lang="ts">
  import SubHeading from "./components/SubHeading.svelte";

  import type { VisualizationSpec } from "vega-embed";
  import { VegaLite } from "svelte-vega";

  import type { PatternForSample } from "./types";

  export let patterns: PatternForSample[];

  const spec: VisualizationSpec = {
    $schema: "https://vega.github.io/schema/vega-lite/v5.json",
    mark: "circle",
    data: {
      name: "table",
    },
    encoding: {
      x: {
        field: "patternId",
      },
      y: {
        field: "label",
      },
      size: { aggregate: "count" },
    },
  };

  $: data = {
    table: patterns,
  };
</script>

<div class="flex flex-col">
  <SubHeading heading={"Distribution"} />
  <VegaLite {data} {spec} />
</div>
