<script lang="ts">
  import ImageComparePatterns from "./ImagePatterns.svelte";
  import LoadingIndicator from "../elements/LoadingIndicator.svelte";
  import IconButton from "../elements/IconButton.svelte";

  import { faTimes } from "@fortawesome/free-solid-svg-icons/faTimes";
  import Fa from "svelte-fa";
  import Select from "svelte-select";

  import { fetchPatternsForImages } from "../api";
  import { imageFilter, selectedPage } from "../stores";

  import type { Pattern } from "../types";

  let width: number;
  let model: string = undefined;

  async function getPatterns(): Promise<Pattern[]> {
    const patterns = await fetchPatternsForImages($imageFilter);
    model = patterns[0].samples[0].model;
    return patterns;
  }
</script>

{#await getPatterns()}
  <div class="h-32 w-full" />
  <LoadingIndicator />
{:then patterns}
  <div class="flex flex-col w-full h-full min-h-0 p-2">
    <div class="flex w-full">
      <div class="grow min-w-0">
        <Select
          placeholder="Model"
          items={[
            ...new Set(patterns.map((pattern) => pattern.samples[0].model)),
          ]}
          value={patterns[0].samples[0].model}
          on:select={(event) => (model = event.detail.value)}
          on:clear={() => (model = undefined)}
        />
      </div>
      <div class="pl-2">
        <IconButton
          filled={true}
          fullHeight={true}
          textColor="text-text-dark"
          on:click={() => {
            imageFilter.set([]);
            selectedPage.set("Overview");
          }}
        >
          <p slot="text" class="whitespace-nowrap">Deselect All Images</p>
          <Fa icon={faTimes} slot="icon" />
        </IconButton>
      </div>
    </div>
    <div class="flex min-w-0 overflow-x-auto pt-2" bind:clientWidth={width}>
      {#if model !== undefined}
        <ImageComparePatterns {patterns} {width} {model} />
      {/if}
    </div>
  </div>
{/await}
