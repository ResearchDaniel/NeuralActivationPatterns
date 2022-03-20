<script lang="ts">
  import type { PatternForSample } from "../../types";
  import PatternImageList from "./PatternImageList.svelte";

  export let samples: PatternForSample[];
  export let model: string;
  export let layer: string;
</script>

<div class="flex flex-col min-h-0">
  <p>Samples ({samples.length})</p>
  <PatternImageList {model} {samples} {layer} />
</div>
