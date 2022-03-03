<script lang="ts">
  import ImageTooltip from "./components/ImageTooltip.svelte";
  import Patterns from "./patterns/Patterns.svelte";

  import type { PatternForSample } from "./types";

  export let model: string;
  export let layer: string;
  export let patterns: PatternForSample[];
</script>

<div class="flex flex-col flex-grow p-2 min-h-0">
  <Patterns {model} {layer} {patterns} />
  <ImageTooltip />
</div>
