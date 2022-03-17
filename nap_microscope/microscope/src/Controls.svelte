<script lang="ts">
  import Dropdown from "./elements/Dropdown.svelte";
  import LabeledComponent from "./elements/LabeledComponent.svelte";
  import SubHeading from "./elements/SubHeading.svelte";

  import {
    fetchFilterMethods,
    fetchFilters,
    fetchLayers,
    fetchModels,
    fetchPatterns,
  } from "./api";
  import type { Patterns } from "./types";

  export let patternsRequest: Promise<Patterns> = undefined;

  let model: string = undefined;
  let layer: string = undefined;
  let filter: string = "---";
  let filterMethod: string = undefined;

  $: patternsRequest = fetchPatterns(model, layer, filter, filterMethod);
</script>

<div class="flex flex-col">
  <SubHeading heading={"Configuration"} />
  {#await fetchModels() then models}
    <div class="pt-2">
      <LabeledComponent name={"Model"}>
        <Dropdown
          items={models}
          bind:value={model}
          on:change={() => {
            layer = undefined;
            filter = "---";
            filterMethod = undefined;
          }}
        />
      </LabeledComponent>
    </div>
  {/await}
  {#if model !== undefined}
    {#await fetchLayers(model) then layers}
      <div class="pt-2">
        <LabeledComponent name={"Layer"}>
          <Dropdown items={layers} bind:value={layer} />
        </LabeledComponent>
      </div>
    {/await}
  {/if}
  {#if model !== undefined && layer !== undefined}
    {#await fetchFilterMethods(model, layer) then filterMethods}
      {#if filterMethods.length > 0}
        <div class="pt-2">
          <LabeledComponent name={"Filter Method"}>
            <Dropdown items={filterMethods} bind:value={filterMethod} />
          </LabeledComponent>
        </div>
      {/if}
    {/await}
    {#if filterMethod !== undefined}
      {#await fetchFilters(model, layer, filterMethod) then filters}
        <div class="pt-2">
          <LabeledComponent name={"Filter"}>
            <Dropdown items={filters} bind:value={filter} />
          </LabeledComponent>
        </div>
      {/await}
    {/if}
  {/if}
</div>
