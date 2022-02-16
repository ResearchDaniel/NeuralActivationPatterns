<script lang="ts">
  import SubSubHeading from "../components/SubSubHeading.svelte";
  import type { PatternForSample } from "../types";
  import PatternImage from "./PatternImage.svelte";
  import PatternImageList from "./PatternImageList.svelte";

  export let samples: PatternForSample[];
  export let patternId: number;
  export let model: string;
  export let layer: string;

  const fetchCenters = (async () => {
    const response = await fetch(
      `/api/get_centers/${model}/${layer}/${patternId}`
    );
    const jsonResponse = await response.json();
    return jsonResponse;
  })();
  const fetchOutliers = (async () => {
    const response = await fetch(
      `/api/get_outliers/${model}/${layer}/${patternId}`
    );
    const jsonResponse = await response.json();
    return jsonResponse;
  })();
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
    {#await fetchCenters}
      Fetching centers.
    {:then centers}
      <div class="flex flex-col pl-4 ">
        <p>Centers</p>
        <PatternImageList {model} indices={centers} {layer} />
      </div>
    {/await}
    {#await fetchOutliers}
      Fetching outliers.
    {:then outliers}
      <div class="flex flex-col pl-4 ">
        <p>Outliers</p>
        <PatternImageList {model} indices={outliers} {layer} />
      </div>
    {/await}
  </div>
</div>
