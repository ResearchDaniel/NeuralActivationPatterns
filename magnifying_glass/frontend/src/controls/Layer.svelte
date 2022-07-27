<script lang="ts">
  import { fetchFeatureVisExists } from "../api";
  import { layerWidth, layerHeight, model, layer } from "../stores";

  export let currentLayer: string;
  export let maxActivatingRequest: Promise<string[]>;
</script>

<div
  style={`width: ${$layerWidth}px; height: ${$layerHeight}px; background: ${
    $layer === currentLayer ? "#0071e3" : "black"
  }`}
  class="rounded flex justify-center items-center shrink-0"
  on:click={() => {
    maxActivatingRequest = undefined;
    layer.set(currentLayer);
  }}
>
  {#await fetchFeatureVisExists($model, currentLayer) then exists}
    {#if exists}
      <img
        class="p-1"
        src={`/api/get_layer_feature_vis/${$model}/${currentLayer}`}
        alt="Layer Level Feature Visualization"
      />
    {/if}
  {/await}
</div>
