<script lang="ts">
  import SubSubHeading from "../components/SubSubHeading.svelte";
  import { numCenters, numOutliers } from "../stores";
  import type { PatternForSample } from "../types";
  import PatternImage from "./PatternImage.svelte";
  import PatternImageList from "./PatternImageList.svelte";
  import AllPatternImages from "./AllPatternImages.svelte";
  import Fa from "svelte-fa";
  import { faExpand } from "@fortawesome/free-solid-svg-icons/faExpand";
  import { faCompress } from "@fortawesome/free-solid-svg-icons/faCompress";
  import { createEventDispatcher } from "svelte";
  import IconButton from "../components/IconButton.svelte";

  export let samples: PatternForSample[];
  export let patternId: number;
  export let model: string;
  export let layer: string;
  export let expanded: boolean = false;

  const dispatch = createEventDispatcher();

  $: sortedSamples = samples.sort(
    (a: PatternForSample, b: PatternForSample) => b.probability - a.probability
  );
  $: centers = sortedSamples.slice(0, $numCenters);
  $: outliers = sortedSamples.slice(-$numOutliers);

  function forwardZoomClicked() {
    dispatch("zoom");
  }
</script>

<div class="flex flex-col mt-4 p-2 border-midgrey border rounded-md">
  <div class="flex">
    <SubSubHeading heading={`ID: ${patternId}`} />
    <SubSubHeading heading={`Size: ${samples.length}`} />
    <div class="ml-auto">
      <IconButton on:click={forwardZoomClicked} plain={true}>
        <Fa icon={expanded ? faCompress : faExpand} slot="icon" />
      </IconButton>
    </div>
  </div>
  <div class="flex flex-col">
    <div class="flex">
      <div class="flex flex-col">
        <p>Average</p>
        <PatternImage
          imagePath={`/api/get_average/${model}/${layer}/${patternId}`}
        />
      </div>
      <div class="flex flex-col pl-4 ">
        <p>Centers</p>
        <PatternImageList {model} samples={centers} {layer} />
      </div>
      <div class="flex flex-col pl-4 ">
        <p>Outliers</p>
        <PatternImageList {model} samples={outliers} {layer} />
      </div>
    </div>
    {#if expanded}
      <AllPatternImages {samples} {model} {layer} />
    {/if}
  </div>
</div>
