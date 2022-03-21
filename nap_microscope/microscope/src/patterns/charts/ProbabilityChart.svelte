<script lang="ts">
  import type { VegaLiteSpec } from "svelte-vega";
  import type { EmbedOptions } from "vega-embed";
  import { VegaLite } from "svelte-vega";

  import type { PatternForSample } from "../../types";

  import { themeConfig } from "../../constants";

  export let samples: PatternForSample[];

  const options = {
    config: themeConfig,
    actions: false,
  } as EmbedOptions;

  $: extent = [
    samples[samples.length - 1].probability === 1
      ? 0
      : samples[samples.length - 1].probability,
    1,
  ];
  $: probabilityHistogramSpec = {
    $schema: "https://vega.github.io/schema/vega-lite/v5.json",
    data: { values: samples },
    width: 100,
    height: 100,
    mark: { type: "bar", tooltip: true },
    encoding: {
      x: {
        bin: { extent: extent },
        field: "probability",
      },
      y: { aggregate: "count", title: "samples" },
    },
  } as VegaLiteSpec;
</script>

<VegaLite spec={probabilityHistogramSpec} {options} />
