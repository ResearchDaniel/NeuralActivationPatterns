<script lang="ts">
  import PatternImage from "./PatternImage.svelte";

  export let indices: number[];
  export let model: string;
  export let layer: number;
</script>

<div class="flex">
  {#each indices as index, i}
    <div class={i + 1 !== indices.length ? "pr-2" : ""}>
      <PatternImage
        imagePath={`/api/get_image/${model}/${index}`}
        {index}
        {layer}
      />
    </div>
  {/each}
</div>
