<script lang="ts">
  import Header from "./header/Header.svelte";
  import Main from "./Main.svelte";
  import Distribution from "./Distribution.svelte";
  import Controls from "./Controls.svelte";
  import ImageTooltip from "./components/ImageTooltip.svelte";
  import PatternCompare from "./patterns/PatternCompare.svelte";
  import Filters from "./Filters.svelte";
  import LoadingIndicator from "./components/LoadingIndicator.svelte";

  import type { PatternForSample, Patterns } from "./types";
  import { labelFilter, predictionFilter, selectedPage } from "./stores";

  let model: string = undefined;
  let layer: string = undefined;
  let layers: string[] = [];
  let dataset: {
    file_name: string;
    label?: string;
    prediction?: string;
  }[] = [];
  const fetchModels = (async () => {
    const response = await fetch(`/api/get_models`);
    const jsonResponse = await response.json();
    const models = jsonResponse["networks"] as string[];
    return models;
  })();

  $: if (model !== undefined) {
    fetch(`/api/get_layers/${model}`)
      .then((response) => response.json())
      .then((jsonResponse) => {
        layers = jsonResponse["layers"] as string[];
      });
    fetch(`/api/get_dataset/${model}`)
      .then((response) => response.json())
      .then((jsonResponse) => {
        dataset = JSON.parse(jsonResponse);
      });
  }
  $: fetchPatterns = (async () => {
    if (dataset.length === 0 || model === undefined || layer === undefined)
      return { samples: [], persistence: [] };
    const infoResponse = await fetch(`/api/get_pattern_info/${model}/${layer}`);
    const infoJsonResponse = await infoResponse.json();
    const info = JSON.parse(infoJsonResponse);
    console.log(info);
    const response = await fetch(`/api/get_patterns/${model}/${layer}`);
    const jsonResponse = await response.json();
    const patterns = JSON.parse(jsonResponse);
    if (patterns.length !== dataset.length)
      return { samples: [], persistence: [] };
    return {
      samples: patterns
        .map((pattern, index) => {
          return {
            patternUid: `${model}_${layer}_${pattern.patternId}`,
            model: model,
            layer: layer,
            patternId: pattern.patternId,
            probability: pattern.probability,
            outlierScore: pattern.outlier_score,
            fileName: dataset[index].file_name,
            label: dataset[index].label,
            prediction: dataset[index].prediction,
          } as PatternForSample;
        })
        .filter((pattern) => pattern.patternId >= 0),
      persistence: info.map((infoElement) => infoElement.pattern_persistence),
    } as Patterns;
  })();
</script>

<main class="h-full">
  <div class="flex flex-col" style="height: 100%;">
    <Header />
    {#if $selectedPage === "Overview"}
      {#await fetchModels then models}
        <div class="flex flex-row p-2 h-96">
          <Controls bind:layers bind:layer bind:model bind:dataset {models} />
          {#await fetchPatterns then patterns}
            <Distribution patterns={patterns.samples} />
          {/await}
        </div>
        {#if layer !== undefined && dataset.length !== 0}
          {#await fetchPatterns}
            <LoadingIndicator />
          {:then patterns}
            <Main {patterns} />
          {/await}
        {/if}
      {/await}
    {:else}
      <PatternCompare />
    {/if}
    {#if $labelFilter.length > 0 || $predictionFilter.length > 0}
      <Filters />
    {/if}
  </div>
  <ImageTooltip />
</main>

<style global lang="postcss">
  @tailwind base;
  @tailwind components;
  @tailwind utilities;
</style>
