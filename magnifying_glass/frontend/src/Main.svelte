<script lang="ts">
  import Explainability from "./explainability/Explainability.svelte";
  import PatternsList from "./patterns/Patterns.svelte";

  import {
    layer,
    model,
    patternFilter,
    patternsWidth,
    showFeatureVis,
  } from "./stores";
  import { fetchFeatureVisExists } from "./api";

  import type { PatternForSample, Patterns } from "./types";

  export let patterns: Patterns;
  export let maxActivating: string[];

  $: featureVisRequest = fetchFeatureVisExists($model, $layer);
  $: sortedSamples = patterns.samples.sort(
    (a: PatternForSample, b: PatternForSample) => b.probability - a.probability
  );
  $: filteredSamples =
    $patternFilter.length === 0
      ? sortedSamples
      : sortedSamples.filter((pattern) => {
          for (let filter of $patternFilter) {
            if (
              pattern.label === filter.label &&
              pattern.patternId === filter.patternId
            ) {
              return true;
            }
          }
          return false;
        });

  let isMouseDown = false;

  function handleMouseDown() {
    isMouseDown = true;
  }

  function handleMouseUp() {
    isMouseDown = false;
  }

  function handleMouseMove(e) {
    if (isMouseDown) {
      patternsWidth.set(
        Math.max(
          200,
          Math.min(window.innerWidth - e.clientX, window.innerWidth - 200)
        )
      );
    }
  }
</script>

<svelte:window on:mousemove={handleMouseMove} on:mouseup={handleMouseUp} />
<div
  class="flex flex-row pl-2 pr-2 min-h-0"
  class:select-none={isMouseDown}
  class:cursor-col-resize={isMouseDown}
>
  <div class="flex flex-1 min-h-0">
    <PatternsList
      {patterns}
      {filteredSamples}
      persistence={patterns.persistence}
    />
  </div>
  {#if $showFeatureVis}
    {#await featureVisRequest then}
      <div
        class="h-full w-1 bg-grey ml-2 mr-2 rounded cursor-col-resize"
        on:mousedown={handleMouseDown}
      />
      <Explainability {maxActivating} featureVis={true} />
    {:catch}
      {#if maxActivating.length > 0}
        <div
          class="h-full w-1 bg-grey ml-2 mr-2 rounded cursor-col-resize"
          on:mousedown={handleMouseDown}
        />
        <Explainability {maxActivating} featureVis={false} />
      {/if}
    {/await}
  {:else if maxActivating.length > 0}
    <div
      class="h-full w-1 bg-grey ml-2 mr-2 rounded cursor-col-resize"
      on:mousedown={handleMouseDown}
    />
    <Explainability {maxActivating} featureVis={false} />
  {/if}
</div>
