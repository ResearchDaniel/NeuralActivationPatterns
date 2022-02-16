<script lang="ts">
  import Fa from "svelte-fa";
  import { fade } from "svelte/transition";
  import { faChartBar } from "@fortawesome/free-solid-svg-icons/faChartBar";
  import { faTag } from "@fortawesome/free-solid-svg-icons/faTag";

  import { tooltip } from "../stores";
  import type { PatternForSample } from "../types";

  export let patterns: PatternForSample[];
  export let labels: string[] | number[];

  let windowWidth = 0;
  let windowHeight = 0;
  let tooltipWidth = 0;
  let tooltipHeight = 0;
  $: yPos =
    windowHeight > $tooltip.mousePos.y + tooltipHeight
      ? $tooltip.mousePos.y
      : $tooltip.mousePos.y - tooltipHeight - 20;
  $: xStyle =
    windowWidth > $tooltip.mousePos.x + tooltipWidth + 10
      ? `left: ${$tooltip.mousePos.x}px;`
      : `right: 10px;`;
  $: style = `top: ${yPos}px; ${xStyle}; background: black`;
</script>

<svelte:window bind:innerWidth={windowWidth} bind:innerHeight={windowHeight} />
{#if $tooltip.hover && $tooltip.layer !== undefined && $tooltip.index !== undefined}
  <div
    class="text-sm fixed p-1 rounded text-text-dark shadow z-10 flex mt-4 mb-4"
    {style}
    transition:fade
    bind:offsetWidth={tooltipWidth}
    bind:offsetHeight={tooltipHeight}
  >
    <div class="flex flex-col p-2 break-all">
      <div class="flex items-center">
        <Fa icon={faTag} class="pr-2" />
        <p>{labels[$tooltip.index]}</p>
      </div>
      <div class="flex items-center">
        <Fa icon={faChartBar} class="pr-2" />
        <p>{patterns[$tooltip.index].probability}</p>
      </div>
    </div>
  </div>
{/if}
