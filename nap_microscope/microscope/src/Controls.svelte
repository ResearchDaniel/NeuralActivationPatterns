<script lang="ts">
  import Select from "svelte-select";

  import { fetchLayers, fetchModels, fetchPatterns } from "./api";
  import { model, layer } from "./stores";
  import type { Patterns } from "./types";

  export let patternsRequest: Promise<Patterns> = undefined;

  $: patternsRequest = fetchPatterns($model, $layer);

  function getSelectedModel(models: string[]): string | undefined {
    if ($model !== undefined) {
      if (!models.includes($model)) {
        model.set(undefined);
      } else {
        return $model;
      }
    }
    if ($model === undefined && models.length > 0) {
      model.set(models[0]);
      return $model;
    }
    return undefined;
  }

  function getSelectedLayer(layers: string[]): string | undefined {
    if ($layer !== undefined) {
      if (!layers.includes($layer)) {
        layer.set(undefined);
      } else {
        return $layer;
      }
    }
    if ($layer === undefined && layers.length > 0) {
      layer.set(layers[0]);
      return $layer;
    }
    return undefined;
  }
</script>

<div class="flex flex-col w-full">
  {#await fetchModels() then models}
    <div class="pt-2">
      <Select
        placeholder="Model"
        items={models}
        value={getSelectedModel(models)}
        on:select={(event) => {
          model.set(event.detail.value);
          layer.set(undefined);
        }}
        on:clear={() => {
          model.set(undefined);
          layer.set(undefined);
        }}
      />
    </div>
  {/await}
  {#if $model !== undefined}
    {#await fetchLayers($model) then layers}
      <div class="pt-2">
        <Select
          placeholder="Layer"
          items={layers}
          value={getSelectedLayer(layers)}
          on:select={(event) => layer.set(event.detail.value)}
          on:clear={() => layer.set(undefined)}
        />
      </div>
    {/await}
  {/if}
</div>
