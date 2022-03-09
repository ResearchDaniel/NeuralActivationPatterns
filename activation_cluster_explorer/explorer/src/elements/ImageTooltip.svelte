<script lang="ts">
  import Fa from "svelte-fa";
  import { fade } from "svelte/transition";
  import { faChartBar } from "@fortawesome/free-solid-svg-icons/faChartBar";
  import { faTag } from "@fortawesome/free-solid-svg-icons/faTag";
  import { faSignature } from "@fortawesome/free-solid-svg-icons/faSignature";
  import { faRobot } from "@fortawesome/free-solid-svg-icons/faRobot";

  import { tooltip } from "../stores";

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
  $: style = `top: ${yPos}px; ${xStyle};`;
</script>

<svelte:window bind:innerWidth={windowWidth} bind:innerHeight={windowHeight} />
{#if $tooltip.hover && $tooltip.layer !== undefined && $tooltip.sample !== undefined}
  <div
    class="bg-white text-sm fixed p-1 rounded shadow z-10 flex mt-4 mb-4"
    {style}
    transition:fade
    bind:offsetWidth={tooltipWidth}
    bind:offsetHeight={tooltipHeight}
  >
    <div class="flex flex-col p-2 break-all">
      <div class="flex items-center">
        <Fa icon={faSignature} class="pr-2" />
        <p>{$tooltip.sample.fileName}</p>
      </div>
      {#if $tooltip.sample.label !== undefined}
        <div class="flex items-center">
          <Fa icon={faTag} class="pr-2" />
          <p>{$tooltip.sample.label}</p>
        </div>
      {/if}
      {#if $tooltip.sample.prediction !== undefined}
        <div class="flex items-center">
          <Fa icon={faRobot} class="pr-2" />
          <p>{$tooltip.sample.prediction}</p>
        </div>
      {/if}
      <div class="flex items-center">
        <Fa icon={faChartBar} class="pr-2" />
        <p>{$tooltip.sample.probability}</p>
      </div>
    </div>
  </div>
{/if}
