<script lang="ts">
  import SubSubHeading from "../components/SubSubHeading.svelte";
  import { numCenters, numOutliers } from "../stores";
  import type { PatternForSample } from "../types";
  import PatternImage from "./PatternImage.svelte";
  import PatternImageList from "./PatternImageList.svelte";

  export let samples: PatternForSample[];
  export let patternId: number;
  export let model: string;
  export let layer: string;

  $: sortedSamples = samples.sort(
    (a: PatternForSample, b: PatternForSample) => b.probability - a.probability
  );
  $: centers = sortedSamples.slice(0, $numCenters);
  $: outliers = sortedSamples.slice(-$numOutliers);
</script>

<div class="flex flex-col mt-4 p-2 border-midgrey border rounded-md">
  <div class="flex">
    <SubSubHeading heading={`Pattern ID: ${patternId}`} />
    <SubSubHeading heading={`Size: ${samples.length}`} />
  </div>
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
</div>
