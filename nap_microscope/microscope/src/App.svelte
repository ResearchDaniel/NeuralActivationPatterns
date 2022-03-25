<script lang="ts">
  import { onMount } from "svelte";

  import Main from "./Main.svelte";
  import Distribution from "./Distribution.svelte";
  import Controls from "./Controls.svelte";
  import Filters from "./Filters.svelte";
  import Header from "./Header.svelte";
  import Settings from "./Settings.svelte";
  import ImageTooltip from "./elements/ImageTooltip.svelte";
  import PatternCompare from "./patterns/PatternCompare.svelte";
  import LoadingIndicator from "./elements/LoadingIndicator.svelte";
  import ImageCompare from "./patterns/ImageCompare.svelte";

  import {
    labelFilter,
    predictionFilter,
    selectedPage,
    showDistribution,
  } from "./stores";
  import type { Patterns } from "./types";
  import { setupURLParams } from "./helpers";

  let patternsRequest: Promise<Patterns> = undefined;
  const urlParams = new URLSearchParams(window.location.search);

  onMount(() => {
    setupURLParams(urlParams);
  });
</script>

<main class="h-full">
  <div class="flex flex-col h-full">
    <Header />
    {#if $selectedPage === "Overview"}
      <div class="flex flex-row p-2">
        <Controls bind:patternsRequest />
        {#if patternsRequest !== undefined && $showDistribution}
          {#await patternsRequest then patterns}
            {#if patterns.samples.length > 0}
              <Distribution patterns={patterns.samples} />
            {/if}
          {/await}
        {/if}
      </div>
      {#if patternsRequest !== undefined}
        {#await patternsRequest}
          <LoadingIndicator />
        {:then patterns}
          {#if patterns.samples.length > 0}
            <Main {patterns} />
          {/if}
        {/await}
      {/if}
    {:else if $selectedPage === "Compare"}
      <PatternCompare />
    {:else}
      <ImageCompare />
    {/if}
    {#if $labelFilter.length > 0 || $predictionFilter.length > 0}
      <Filters />
    {/if}
  </div>
  <ImageTooltip />
  <Settings />
</main>

<style global lang="postcss">
  @tailwind base;
  @tailwind elements;
  @tailwind utilities;
</style>
