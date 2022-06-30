<script lang="ts">
  import Select from "svelte-select";
  import Network from "./Network.svelte";

  import {
    fetchLayers,
    fetchModels,
    fetchPatterns,
    fetchMaxActivating,
  } from "../api";
  import { model, layer, showMaxActivating } from "../stores";
  import type { Patterns } from "../types";

  export let patternsRequest: Promise<Patterns> = undefined;
  export let maxActivatingRequest: Promise<string[]> = undefined;

  $: patternsRequest = fetchPatterns($model, $layer);
  $: maxActivatingRequest = fetchMaxActivating(
    $model,
    $layer,
    $showMaxActivating
  );

  function getSelectedModel(models: string[]): string | undefined {
    if ($model !== undefined) {
      return $model;
    }
    if ($model === undefined && models.length > 0) {
      model.set(models[0]);
      return $model;
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
      <Network {layers} />
    {/await}
  {/if}
</div>
