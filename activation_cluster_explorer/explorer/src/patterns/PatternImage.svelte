<script lang="ts">
  import { tooltip } from "../stores";
  import type { PatternForSample } from "../types";

  export let imagePath: string;
  export let sample: PatternForSample = undefined;
  export let layer: number = undefined;

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
</script>

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
