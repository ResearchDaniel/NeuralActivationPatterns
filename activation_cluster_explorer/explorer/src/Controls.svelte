<script lang="ts">
  import Dropdown from "./components/Dropdown.svelte";
  import LabeledComponent from "./components/LabeledComponent.svelte";
  import SubHeading from "./components/SubHeading.svelte";

  export let models: string[];
  export let model: string;
  export let layers: string[];
  export let layer: string;
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
          layer = undefined;
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
</div>
