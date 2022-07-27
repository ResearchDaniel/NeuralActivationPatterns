<script lang="ts">
  import Layer from "./Layer.svelte";
  import LayerDivider from "../elements/LayerDivider.svelte";

  import { layer } from "../stores";
  import LayerDescription from "./LayerDescription.svelte";

  export let layers: string[];
  export let maxActivatingRequest: Promise<string[]>;

  $: getSelectedLayer(layers);

  function getSelectedLayer(layers: string[]): string | undefined {
    if ($layer !== undefined) {
      return $layer;
    }
    if ($layer === undefined && layers.length > 0) {
      layer.set(layers[0]);
      return $layer;
    }
    return undefined;
  }
</script>

<div class="flex flex-col py-2 overflow-x-auto overflow-y-hidden">
  <div class="flex items-center">
    {#each layers as currentLayer, index}
      <Layer {currentLayer} {maxActivatingRequest} />
      {#if index < layers.length - 1}
        <LayerDivider />
      {/if}
    {/each}
  </div>
  <div class="flex items-center">
    {#each layers as currentLayer, index}
      <LayerDescription {currentLayer} />
      {#if index < layers.length - 1}
        <LayerDivider transparent={true} />
      {/if}
    {/each}
  </div>
</div>
