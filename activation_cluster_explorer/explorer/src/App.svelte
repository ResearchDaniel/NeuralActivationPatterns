<script lang="ts">
  import Main from "./Main.svelte";
  import Distribution from "./Distribution.svelte";
  import Controls from "./Controls.svelte";
  import Filters from "./Filters.svelte";
  import Header from "./Header.svelte";
  import Settings from "./Settings.svelte";
  import ImageTooltip from "./elements/ImageTooltip.svelte";
  import PatternCompare from "./patterns/PatternCompare.svelte";
  import LoadingIndicator from "./elements/LoadingIndicator.svelte";

  import {
    imageFilter,
    labelFilter,
    predictionFilter,
    selectedPage,
  } from "./stores";
  import type { Patterns } from "./types";

  let patternsRequest: Promise<Patterns> = undefined;
</script>

<main class="h-full">
  <div class="flex flex-col h-full">
    <Header />
    {#if $selectedPage === "Overview"}
      <div class="flex flex-row p-2 h-96">
        <Controls bind:patternsRequest />
        {#if patternsRequest !== undefined}
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
          <Main {patterns} />
        {/await}
      {/if}
    {:else}
      <PatternCompare />
    {/if}
    {#if $labelFilter.length > 0 || $predictionFilter.length > 0 || $imageFilter.length > 0}
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
