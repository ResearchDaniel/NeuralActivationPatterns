<script lang="ts">
  import MaxActivations from "./MaxActivations.svelte";
  import FeatureVis from "./FeatureVis.svelte";

  import { patternsWidth } from "../stores";

  export let maxActivating: string[];
  export let featureVis: boolean;
</script>

<div class="flex flex-col min-h-0 h-full" style={`width: ${$patternsWidth}px;`}>
  {#if maxActivating.length > 0}
    <MaxActivations {maxActivating} />
  {/if}
  {#if featureVis}
    <FeatureVis />
  {/if}
</div>
