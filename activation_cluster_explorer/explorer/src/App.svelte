<script lang="ts">
  import Main from "./Main.svelte";
  import Distribution from "./Distribution.svelte";
  import Controls from "./Controls.svelte";
  import Filters from "./Filters.svelte";
  import Header from "./header/Header.svelte";
  import ImageTooltip from "./components/ImageTooltip.svelte";
  import PatternCompare from "./patterns/PatternCompare.svelte";
  import LoadingIndicator from "./components/LoadingIndicator.svelte";

  import type { PatternForSample, Patterns } from "./types";
  import { labelFilter, predictionFilter, selectedPage } from "./stores";

  let model: string = undefined;
  let layer: string = undefined;
  let labels: Record<number, string> = undefined;
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
    fetch(`/api/get_labels/${model}`)
      .then((response) => response.json())
      .then((jsonResponse) => (labels = jsonResponse));
  }
  $: fetchPatterns = (async () => {
    if (
      dataset.length === 0 ||
      model === undefined ||
      layer === undefined ||
      labels === undefined
    )
      return { samples: [], persistence: [] };
    const infoResponse = await fetch(`/api/get_pattern_info/${model}/${layer}`);
    const infoJsonResponse = await infoResponse.json();
    const info = JSON.parse(infoJsonResponse);
    const response = await fetch(`/api/get_patterns/${model}/${layer}`);
    const jsonResponse = await response.json();
    const patterns = JSON.parse(jsonResponse);
    if (patterns.length !== dataset.length)
      return { samples: [], persistence: [] };
    return {
      samples: patterns
        .map(
          (
            pattern: {
              patternId: number;
              probability: number;
              outlier_score: number;
            },
            index: number
          ) => {
            return {
              patternUid: `${model}_${layer}_${pattern.patternId}`,
              model: model,
              layer: layer,
              patternId: pattern.patternId,
              probability: pattern.probability,
              outlierScore: pattern.outlier_score,
              fileName: dataset[index].file_name,
              labelIndex: dataset[index].label,
              label: labels[dataset[index].label],
              predictionIndex: dataset[index].prediction,
              prediction: labels[dataset[index].prediction],
            } as PatternForSample;
          }
        )
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
          <Controls
            bind:layers
            bind:layer
            bind:model
            bind:dataset
            bind:labels
            {models}
          />
          {#await fetchPatterns then patterns}
            {#if patterns.samples.length > 0}
              <Distribution patterns={patterns.samples} />
            {/if}
          {/await}
        </div>
        {#if layer !== undefined && labels !== undefined && dataset.length !== 0}
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
