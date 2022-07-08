<script lang="ts">
  import MaxActivations from "./MaxActivations.svelte";
  import FeatureVis from "./FeatureVis.svelte";

  import { fetchFeatureVisExists } from "../api";
  import { layer, model } from "../stores";

  export let maxActivating: string[];

  $: featureVisRequest = fetchFeatureVisExists($model, $layer);
</script>

<div class="flex flex-1 flex-col min-h-0 h-full">
  {#await featureVisRequest then}
    <MaxActivations {maxActivating} />
    <FeatureVis />
  {:catch}
    <MaxActivations {maxActivating} />
  {/await}
</div>
