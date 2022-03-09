<script lang="ts">
  import SubHeading from "./components/SubHeading.svelte";

  import type { VisualizationSpec } from "vega-embed";
  import { VegaLite } from "svelte-vega";
  import type { EmbedOptions } from "vega-embed";

  import type { PatternForSample } from "./types";
  import { themeConfig } from "./constants";
  import { patternFilter } from "./stores";

  export let patterns: PatternForSample[];

  const options = {
    actions: false,
    config: themeConfig,
  } as EmbedOptions;

  $: spec = {
    $schema: "https://vega.github.io/schema/vega-lite/v5.json",
    mark: { type: "circle", tooltip: true },
    data: { values: patterns },
    params: [
      { name: "select", select: { type: "point", encodings: ["x", "y"] } },
    ],
    encoding: {
      x: { field: "label" },
      y: { field: "patternId" },
      size: { aggregate: "count" },
      fillOpacity: {
        condition: { param: "select", value: 1 },
        value: 0.3,
      },
    },
  } as VisualizationSpec;

  function handleSelection(...args: any) {
    if (args[1].vlPoint !== undefined) {
      patternFilter.set(args[1].vlPoint.or);
    } else {
      patternFilter.set([]);
    }
  }
</script>

<div class="flex flex-col">
  <SubHeading heading={"Distribution"} />
  <div class="overflow-auto">
    <VegaLite {spec} {options} signalListeners={{ select: handleSelection }} />
  </div>
</div>
