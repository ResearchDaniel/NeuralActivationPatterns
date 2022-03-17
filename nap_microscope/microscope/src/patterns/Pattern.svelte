<script lang="ts">
  import { imageFilter, pinnedPatterns, selectedPage } from "../stores";

  import Fa from "svelte-fa";
  import FaLayers from "svelte-fa/src/fa-layers.svelte";
  import { faThumbtack } from "@fortawesome/free-solid-svg-icons/faThumbtack";
  import { faSlash } from "@fortawesome/free-solid-svg-icons/faSlash";
  import { faFilter } from "@fortawesome/free-solid-svg-icons/faFilter";

  import SubSubHeading from "../elements/SubSubHeading.svelte";
  import AllPatternImages from "./AllPatternImages.svelte";
  import IconButton from "../elements/IconButton.svelte";
  import PatternOverview from "./PatternOverview.svelte";

  import type { PatternForSample } from "../types";

  export let samples: PatternForSample[];
  export let filteredSamples: PatternForSample[];
  export let expanded: boolean = false;

  $: uid = samples[0].patternUid;
  $: patternId = samples[0].patternId;
  $: model = samples[0].model;
  $: layer = samples[0].layer;

  function unpinPattern() {
    pinnedPatterns.update((patterns) => {
      delete patterns[uid];
      if (Object.keys(patterns).length === 0) {
        selectedPage.set("Overview");
      }
      return patterns;
    });
  }

  function pinPattern() {
    $pinnedPatterns[uid] = samples;
    pinnedPatterns.set({ ...$pinnedPatterns });
  }

  function filterAll() {
    const notFiltered = filteredSamples.filter(
      (sample) => !$imageFilter.includes(sample.fileName)
    );
    if (notFiltered.length === 0) {
      imageFilter.update((filters) => {
        filteredSamples.forEach((sample) => {
          const index = filters.indexOf(sample.fileName, 0);
          if (index > -1) {
            filters.splice(index, 1);
          }
        });
        return filters;
      });
    } else {
      const fileNames = notFiltered.map((item) => item.fileName);
      imageFilter.update((filters) => [...new Set([...filters, ...fileNames])]);
    }
  }
</script>

<div
  class="flex flex-col box-shadow-xl p-2 border-grey border rounded-md {expanded
    ? 'min-h-0 m-2 min-w-compare'
    : 'w-full mb-2'}"
>
  <div class="flex">
    <div class="flex flex-wrap">
      <SubSubHeading heading={`ID: ${patternId}`} />
      <SubSubHeading heading={`Size: ${samples.length}`} />
      {#if expanded}
        <SubSubHeading heading={`Model: ${model}`} />
        <SubSubHeading heading={`Layer: ${layer}`} />
        {#if samples[0].filter !== undefined}
          <SubSubHeading heading={`Filter: ${samples[0].filter}`} />
        {/if}
      {/if}
    </div>
    <div class="ml-auto">
      <IconButton on:click={filterAll}>
        <Fa icon={faFilter} slot="icon" />
      </IconButton>
      {#if $pinnedPatterns[uid] !== undefined}
        <IconButton on:click={unpinPattern}>
          <FaLayers slot="icon">
            <Fa icon={faThumbtack} />
            <Fa icon={faSlash} />
          </FaLayers>
        </IconButton>
      {:else}
        <IconButton on:click={pinPattern}>
          <Fa icon={faThumbtack} slot="icon" />
        </IconButton>
      {/if}
    </div>
  </div>
  <div class="flex flex-col" class:min-h-0={expanded}>
    <PatternOverview
      {samples}
      {model}
      {layer}
      {patternId}
      {filteredSamples}
      {expanded}
    />
    {#if expanded}
      <AllPatternImages samples={filteredSamples} {model} {layer} />
    {/if}
  </div>
</div>
