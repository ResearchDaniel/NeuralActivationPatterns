<script lang="ts">
  import MaxActivations from "./MaxActivations.svelte";

  import { patternsWidth } from "../stores";

  export let maxActivating: string[];
</script>

<div class="flex flex-col min-h-0 h-full" style={`width: ${$patternsWidth}px;`}>
  <MaxActivations {maxActivating} />
</div>
