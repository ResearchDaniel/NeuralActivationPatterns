<script lang="ts">
  import {
    compactPatterns,
    imageFilter,
    pinnedPatterns,
    selectedPage,
  } from "../stores";

  import Fa from "svelte-fa";
  import FaLayers from "svelte-fa/src/fa-layers.svelte";
  import { faThumbtack } from "@fortawesome/free-solid-svg-icons/faThumbtack";
  import { faSlash } from "@fortawesome/free-solid-svg-icons/faSlash";
  import { faFilter } from "@fortawesome/free-solid-svg-icons/faFilter";

  import SubSubHeading from "../elements/SubSubHeading.svelte";
  import AllPatternImages from "./images/AllPatternImages.svelte";
  import IconButton from "../elements/IconButton.svelte";
  import PatternOverview from "./PatternOverview.svelte";

  import type { PatternForSample, Pattern } from "../types";

  export let pattern: Pattern;
  export let filteredSamples: PatternForSample[];
  export let expanded: boolean = false;
  export let patternWidth: number = undefined;
  export let patternIndex: number = undefined;

  const tableau20 = [
    "#4c78a8",
    "#9ecae9",
    "#f58518",
    "#ffbf79",
    "#54a24b",
    "#88d27a",
    "#b79a20",
    "#f2cf5b",
    "#439894",
    "#83bcb6",
    "#e45756",
    "#ff9d98",
    "#79706e",
    "#bab0ac",
    "#d67195",
    "#fcbfd2",
    "#b279a2",
    "#d6a5c9",
    "#9e765f",
    "#d8b5a5",
  ];

  $: uid = pattern.samples[0].patternUid;
  $: patternId = pattern.samples[0].patternId;
  $: model = pattern.samples[0].model;
  $: layer = pattern.samples[0].layer;

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
    $pinnedPatterns[uid] = pattern;
    pinnedPatterns.set({ ...$pinnedPatterns });
  }

  function filterAll() {
    const notFiltered = filteredSamples.filter(
      (sample) =>
        !$imageFilter.some(
          (filter) =>
            sample.fileName === filter.image && sample.model == filter.model
        )
    );
    if (notFiltered.length === 0) {
      imageFilter.update((filters) => {
        filteredSamples.forEach((sample) => {
          const index = $imageFilter.findIndex(
            (filter) =>
              filter.image === sample.fileName && filter.model === sample.model,
            0
          );
          if (index > -1) {
            filters.splice(index, 1);
          }
        });
        return filters;
      });
    } else {
      const toAdd = notFiltered.map((item) => {
        return { image: item.fileName, model: item.model };
      });
      imageFilter.update((filters) => [...filters, ...toAdd]);
    }
  }
</script>

<div
  class="flex flex-col box-shadow-xl p-2 border-grey border rounded-md min-w-0"
  class:min-h-0={expanded}
  class:shrink-0={expanded}
  class:mb-2={!expanded}
  class:w-full={!$compactPatterns && !expanded}
  class:mr-2={$compactPatterns || expanded}
  style={expanded ? `width: ${patternWidth}px` : ""}
>
  <div class="flex items-baseline">
    <div class="flex flex-wrap">
      <SubSubHeading
        heading={`ID: ${patternId}`}
        color={expanded ? tableau20[patternIndex] : undefined}
      />
      <SubSubHeading heading={`Size: ${pattern.samples.length}`} />
      {#if expanded}
        <SubSubHeading heading={`Model: ${model}`} />
        <SubSubHeading heading={`Layer: ${layer}`} />
        {#if pattern.samples[0].filter !== undefined}
          <SubSubHeading heading={`Filter: ${pattern.samples[0].filter}`} />
        {/if}
      {/if}
    </div>
    <div class="flex ml-auto">
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
      {pattern}
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
