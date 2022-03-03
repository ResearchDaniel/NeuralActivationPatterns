<script lang="ts">
  import type { PatternForSample } from "../types";
  import PatternImageList from "./PatternImageList.svelte";

  export let samples: PatternForSample[];
  export let model: string;
  export let layer: string;
</script>

<div class="flex flex-col">
  <p>All Samples</p>
  <PatternImageList {model} {samples} {layer} />
</div>
