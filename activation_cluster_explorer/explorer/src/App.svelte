<script lang="ts">
  import Header from "./header/Header.svelte";
  import Main from "./Main.svelte";
  import Dropdown from "./components/Dropdown.svelte";
  import LabeledComponent from "./components/LabeledComponent.svelte";

  let model: string = undefined;
  let layer: string = undefined;
  let layers: string[] = [];
  const fetchModels = (async () => {
    const response = await fetch(`/api/get_models`);
    const jsonResponse = await response.json();
    const models = jsonResponse["networks"] as string[];
    return models;
  })();

  $: if (model !== undefined) {
    fetch(`/api/get_layers/${model}`)
      .then((response) => response.json())
      .then((jsonResponse) => (layers = jsonResponse["layers"] as string[]));
  }
  $: fetchDataset = (async () => {
    if (model !== undefined) {
      const response = await fetch(`/api/get_dataset/${model}`);
      const jsonResponse = await response.json();
      return JSON.parse(jsonResponse);
    } else {
      return [];
    }
  })();
</script>

<main class="h-full">
  <div class="flex flex-col" style="height: 100%;">
    <Header />
    {#await fetchModels then models}
      <div class="flex flex-row p-2">
        <LabeledComponent name={"Model"}>
          <Dropdown
            items={models}
            bind:value={model}
            on:change={() => {
              layers = [];
              layer = undefined;
            }}
          />
        </LabeledComponent>
        {#if model !== undefined && layers.length !== 0}
          <LabeledComponent name={"Layer"}>
            <Dropdown items={layers} bind:value={layer} />
          </LabeledComponent>
        {/if}
      </div>
      {#await fetchDataset then dataset}
        {#if layer !== undefined}
          <Main {model} {layer} {dataset} />
        {/if}
      {/await}
    {/await}
  </div>
</main>

<style global lang="postcss">
  @tailwind base;
  @tailwind components;
  @tailwind utilities;
</style>
