<script lang="ts">
  import { pinnedPatterns, pinnedPatternUids } from "../stores";

  import Fa from "svelte-fa";
  import FaLayers from "svelte-fa/src/fa-layers.svelte";
  import { faThumbtack } from "@fortawesome/free-solid-svg-icons/faThumbtack";
  import { faSlash } from "@fortawesome/free-solid-svg-icons/faSlash";

  import SubSubHeading from "../components/SubSubHeading.svelte";
  import AllPatternImages from "./AllPatternImages.svelte";
  import IconButton from "../components/IconButton.svelte";
  import PatternOverview from "./PatternOverview.svelte";

  import type { PatternForSample } from "../types";

  export let samples: PatternForSample[];
  export let expanded: boolean = false;

  $: sortedSamples = samples.sort(
    (a: PatternForSample, b: PatternForSample) => b.probability - a.probability
  );
  $: uid = samples[0].patternUid;
  $: patternId = samples[0].patternId;
  $: model = samples[0].model;
  $: layer = samples[0].layer;

  function unpinPattern() {
    pinnedPatterns.update((patterns) => {
      delete patterns[uid];
      return patterns;
    });
  }

  function pinPattern() {
    $pinnedPatterns[uid] = sortedSamples;
    pinnedPatterns.set({ ...$pinnedPatterns });
  }
</script>

<div
  class="flex flex-col p-2 border-midgrey border rounded-md {expanded
    ? 'min-h-0 m-2 min-w-compare'
    : 'w-full mt-4'}"
>
  <div class="flex">
    <SubSubHeading heading={`ID: ${patternId}`} />
    <SubSubHeading heading={`Size: ${samples.length}`} />
    <div class="ml-auto">
      {#if $pinnedPatterns[uid] !== undefined}
        <IconButton on:click={unpinPattern} plain={true}>
          <FaLayers slot="icon">
            <Fa icon={faThumbtack} />
            <Fa icon={faSlash} />
          </FaLayers>
        </IconButton>
      {:else}
        <IconButton on:click={pinPattern} plain={true}>
          <Fa icon={faThumbtack} slot="icon" />
        </IconButton>
      {/if}
    </div>
  </div>
  <div class="flex flex-col" class:min-h-0={expanded}>
    <PatternOverview samples={sortedSamples} {model} {layer} {patternId} />
    {#if expanded}
      <AllPatternImages {samples} {model} {layer} />
    {/if}
  </div>
</div>
