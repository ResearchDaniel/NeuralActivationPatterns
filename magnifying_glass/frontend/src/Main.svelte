<script lang="ts">
  import Explainability from "./explainability/Explainability.svelte";
  import PatternsList from "./patterns/Patterns.svelte";
  import { patternFilter, patternsWidth } from "./stores";

  import type { PatternForSample, Patterns } from "./types";

  export let patterns: Patterns;
  export let maxActivating: string[];

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
        Math.max(200, Math.min(e.clientX, window.innerWidth - 200))
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
  <div class="flex min-h-0" style={`width: ${$patternsWidth}px;`}>
    <PatternsList
      {patterns}
      {filteredSamples}
      persistence={patterns.persistence}
    />
  </div>
  {#if maxActivating.length > 0}
    <div
      class="h-full w-1 bg-grey ml-2 mr-2 rounded cursor-col-resize"
      on:mousedown={handleMouseDown}
    />
    <Explainability {maxActivating} />
  {/if}
</div>
