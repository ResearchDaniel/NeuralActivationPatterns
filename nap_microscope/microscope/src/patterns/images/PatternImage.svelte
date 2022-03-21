<script lang="ts">
  import { imageFilter, tooltip } from "../../stores";
  import Fa from "svelte-fa";
  import FaLayers from "svelte-fa/src/fa-layers.svelte";
  import { faPlay } from "@fortawesome/free-solid-svg-icons/faPlay";
  import { faExclamation } from "@fortawesome/free-solid-svg-icons/faExclamation";
  import type { PatternForSample } from "../../types";

  export let imagePath: string;
  export let sample: PatternForSample = undefined;
  export let layer: string = undefined;

  let hover = false;
  let m = { x: 0, y: 0 };

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

  function addImageFilter() {
    imageFilter.update((filters) => [
      ...new Set([...filters, sample.fileName]),
    ]);
  }
</script>

<div class="relative" on:click={addImageFilter}>
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
  {#if sample !== undefined && sample.label !== undefined && sample.prediction !== undefined && sample.label !== sample.prediction}
    <div class="absolute top-1 right-2">
      <FaLayers>
        <Fa icon={faPlay} rotate={-90} color="#ff453a" />
        <Fa icon={faExclamation} scale={0.5} translateY={0.1} color="white" />
      </FaLayers>
    </div>
  {/if}
</div>
