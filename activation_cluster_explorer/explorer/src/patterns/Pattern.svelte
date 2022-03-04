<script lang="ts">
  import SubSubHeading from "../components/SubSubHeading.svelte";
  import type { PatternForSample } from "../types";
  import AllPatternImages from "./AllPatternImages.svelte";
  import Fa from "svelte-fa";
  import { faExpand } from "@fortawesome/free-solid-svg-icons/faExpand";
  import { faCompress } from "@fortawesome/free-solid-svg-icons/faCompress";
  import { createEventDispatcher } from "svelte";
  import IconButton from "../components/IconButton.svelte";
  import PatternOverview from "./PatternOverview.svelte";

  export let samples: PatternForSample[];
  export let patternId: number;
  export let model: string;
  export let layer: string;
  export let expanded: boolean = false;

  const dispatch = createEventDispatcher();

  $: sortedSamples = samples.sort(
    (a: PatternForSample, b: PatternForSample) => b.probability - a.probability
  );

  function forwardZoomClicked() {
    dispatch("zoom");
  }
</script>

<div
  class="flex flex-col mt-4 p-2 border-midgrey border rounded-md w-full"
  class:min-h-0={expanded}
>
  <div class="flex">
    <SubSubHeading heading={`ID: ${patternId}`} />
    <SubSubHeading heading={`Size: ${samples.length}`} />
    <div class="ml-auto">
      <IconButton on:click={forwardZoomClicked} plain={true}>
        <Fa icon={expanded ? faCompress : faExpand} slot="icon" />
      </IconButton>
    </div>
  </div>
  <div class="flex flex-col" class:min-h-0={expanded}>
    <PatternOverview samples={sortedSamples} {model} {layer} {patternId} />
    {#if expanded}
      <AllPatternImages {samples} {model} {layer} />
    {/if}
  </div>
</div>
