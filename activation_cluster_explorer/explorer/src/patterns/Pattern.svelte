<script lang="ts">
  import { pinnedPatterns, selectedPage } from "../stores";

  import Fa from "svelte-fa";
  import FaLayers from "svelte-fa/src/fa-layers.svelte";
  import { faThumbtack } from "@fortawesome/free-solid-svg-icons/faThumbtack";
  import { faSlash } from "@fortawesome/free-solid-svg-icons/faSlash";

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
</script>

<div
  class="flex flex-col box-shadow-xl p-2 border-grey border rounded-md {expanded
    ? 'min-h-0 m-2 min-w-compare'
    : 'w-full mb-2'}"
>
  <div class="flex">
    <SubSubHeading heading={`ID: ${patternId}`} />
    <SubSubHeading heading={`Size: ${samples.length}`} />
    <div class="ml-auto">
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
    <PatternOverview {samples} {model} {layer} {patternId} {filteredSamples} />
    {#if expanded}
      <AllPatternImages samples={filteredSamples} {model} {layer} />
    {/if}
  </div>
</div>
