<script lang="ts">
  import ImageComparePatterns from "./ImagePatterns.svelte";
  import LoadingIndicator from "../elements/LoadingIndicator.svelte";

  import { fetchPatternsForImages } from "../api";

  import { imageFilter } from "../stores";

  let width: number;
</script>

{#await fetchPatternsForImages($imageFilter)}
  <div class="h-32 w-full" />
  <LoadingIndicator />
{:then patterns}
  <div class="flex min-w-0 overflow-x-auto h-full p-2" bind:clientWidth={width}>
    <ImageComparePatterns {patterns} {width} />
  </div>
{/await}
