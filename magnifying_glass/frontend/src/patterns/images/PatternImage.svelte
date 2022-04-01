<script lang="ts">
  import { imageFilter, selectedPage, tooltip } from "../../stores";
  import Fa from "svelte-fa";
  import FaLayers from "svelte-fa/src/fa-layers.svelte";
  import { faPlay } from "@fortawesome/free-solid-svg-icons/faPlay";
  import { faExclamation } from "@fortawesome/free-solid-svg-icons/faExclamation";
  import { faSquareCheck } from "@fortawesome/free-solid-svg-icons/faSquareCheck";
  import type { PatternForSample } from "../../types";

  export let imagePath: string;
  export let sample: PatternForSample = undefined;
  export let layer: string = undefined;

  let hover = false;
  let m = { x: 0, y: 0 };

  $: selected = $imageFilter.some(
    (filter) =>
      filter.image === sample.fileName && filter.model === sample.model
  );
  $: misclassified =
    sample !== undefined &&
    sample.label !== undefined &&
    sample.prediction !== undefined &&
    sample.label !== sample.prediction;

  function handleMousemove(event: MouseEvent) {
    m.x = event.clientX;
    m.y = event.clientY;
    updateSampleTooltip();
  }

  function updateSampleTooltip() {
    tooltip.set({
      hover: hover,
      mousePos: m,
      sample: sample,
      layer: layer,
    });
  }

  function handleSelectionImage() {
    const index = $imageFilter.findIndex(
      (filter) =>
        filter.image === sample.fileName && filter.model === sample.model,
      0
    );
    if (index > -1) {
      imageFilter.update((filters) => {
        filters.splice(index, 1);
        if (filters.length === 0 && $selectedPage === "Images") {
          selectedPage.set("Overview");
        }
        return filters;
      });
    } else {
      imageFilter.update((filters) => [
        ...new Set([
          ...filters,
          { image: sample.fileName, model: sample.model },
        ]),
      ]);
    }
  }
</script>

<div class="relative" on:click={handleSelectionImage}>
  <img
    class="h-32"
    src={imagePath}
    on:mouseenter={() => {
      if (sample !== undefined && layer !== undefined) {
        hover = true;
        updateSampleTooltip();
      }
    }}
    on:mouseleave={() => {
      if (sample !== undefined && layer !== undefined) {
        hover = false;
        updateSampleTooltip();
      }
    }}
    on:mousemove={handleMousemove}
    alt="Data Sample"
  />
  {#if selected || misclassified}
    <div class="absolute top-1 right-1 flex bg-black_semi p-1 rounded">
      {#if selected}
        <FaLayers>
          <Fa icon={faSquareCheck} color="white" />
        </FaLayers>
      {/if}
      {#if misclassified}
        <FaLayers>
          <Fa icon={faPlay} rotate={-90} color="#ff453a" />
          <Fa icon={faExclamation} scale={0.5} translateY={0.1} color="white" />
        </FaLayers>
      {/if}
    </div>
  {/if}
</div>
