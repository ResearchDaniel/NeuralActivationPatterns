<script lang="ts">
  import { layer, layerWidth, layerHeight } from "../stores";

  export let layers: string[];

  $: getSelectedLayer(layers);

  function getSelectedLayer(layers: string[]): string | undefined {
    if ($layer !== undefined) {
      return $layer;
    }
    if ($layer === undefined && layers.length > 0) {
      layer.set(layers[0]);
      return $layer;
    }
    return undefined;
  }
</script>

<svg width="100%" height={$layerHeight + 50}>
  <g transform="translate(0, 10)">
    {#each layers as currentLayer, index}
      {#if index < layers.length - 1}
        <line
          x1={$layerWidth + index * ($layerWidth + $layerWidth / 2)}
          y1={$layerHeight / 2}
          x2={$layerWidth +
            $layerWidth / 2 +
            index * ($layerWidth + $layerWidth / 2)}
          y2={$layerHeight / 2}
          stroke="black"
        />
      {/if}
      <rect
        x={index * ($layerWidth + $layerWidth / 2)}
        width={$layerWidth}
        height={$layerHeight}
        rx={5}
        fill={$layer === currentLayer ? "#0071e3" : "black"}
        on:click={() => {
          layer.set(currentLayer);
        }}
      />
      <g
        transform={`translate(${
          index * ($layerWidth + $layerWidth / 2) + $layerWidth / 2
        }, ${$layerHeight + 5 + 12})`}
      >
        <text text-anchor="middle" transform="rotate(10)">
          {currentLayer}
        </text>
      </g>
    {/each}
  </g>
</svg>
