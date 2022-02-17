<script lang="ts">
  import { tooltip } from "../stores";

  export let imagePath: string;
  export let index: number = undefined;
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
      index: index,
      layer: layer,
    });
  }
</script>

<img
  class="h-32"
  src={imagePath}
  on:mouseenter={() => {
    if (index !== undefined && layer !== undefined) {
      hover = true;
      updateSampleTooltip();
    }
  }}
  on:mouseleave={() => {
    if (index !== undefined && layer !== undefined) {
      hover = false;
      updateSampleTooltip();
    }
  }}
  on:mousemove={handleMousemove}
  alt="Data Sample"
/>
