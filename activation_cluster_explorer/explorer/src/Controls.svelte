<script lang="ts">
  import Dropdown from "./elements/Dropdown.svelte";
  import LabeledComponent from "./elements/LabeledComponent.svelte";
  import SubHeading from "./elements/SubHeading.svelte";

  export let models: string[];
  export let model: string = undefined;
  export let layers: string[];
  export let layer: string = undefined;
  export let filters: string[];
  export let filter: string = "---";
  export let labels: Record<number, string> = undefined;
  export let dataset: {
    file_name: string;
    label?: string;
    prediction?: string;
  }[];
</script>

<div class="flex flex-col">
  <SubHeading heading={"Configuration"} />
  <div class="pt-2">
    <LabeledComponent name={"Model"}>
      <Dropdown
        items={models}
        bind:value={model}
        on:change={() => {
          layers = [];
          dataset = [];
          filters = [];
          layer = undefined;
          labels = undefined;
          filter = "---";
        }}
      />
    </LabeledComponent>
  </div>
  {#if model !== undefined && layers.length !== 0}
    <div class="pt-2">
      <LabeledComponent name={"Layer"}>
        <Dropdown items={layers} bind:value={layer} />
      </LabeledComponent>
    </div>
  {/if}
  {#if model !== undefined && layer !== undefined && filters.length > 1}
    <div class="pt-2">
      <LabeledComponent name={"Filter"}>
        <Dropdown items={filters} bind:value={filter} />
      </LabeledComponent>
    </div>
  {/if}
</div>
