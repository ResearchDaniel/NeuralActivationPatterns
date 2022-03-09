<script lang="ts">
  import { labelFilter, predictionFilter } from "./stores";
  import Pill from "./elements/Pill.svelte";
  import SubSubHeading from "./elements/SubSubHeading.svelte";
  import IconButton from "./elements/IconButton.svelte";
  import { faTimes } from "@fortawesome/free-solid-svg-icons/faTimes";
  import Fa from "svelte-fa";

  function removeLabelFilter(element: string) {
    labelFilter.update((filters) => {
      const index = filters.indexOf(element, 0);
      if (index > -1) {
        filters.splice(index, 1);
      }
      return filters;
    });
  }
  function removePredictionFilter(element: string) {
    predictionFilter.update((filters) => {
      const index = filters.indexOf(element, 0);
      if (index > -1) {
        filters.splice(index, 1);
      }
      return filters;
    });
  }
</script>

<div class="shadow-top flex items-center p-2 z-10">
  <SubSubHeading heading={"Filters"} />
  <div class="flex items-center overflow-x-auto min-w-0">
    {#if $labelFilter.length > 0}
      <div class="flex items-center ml-4">
        <p>Labels:</p>
        {#each $labelFilter as filter}
          <Pill element={filter} on:remove={() => removeLabelFilter(filter)} />
        {/each}
      </div>
    {/if}
    {#if $predictionFilter.length > 0}
      <div class="flex items-center ml-4">
        <p>Predictions:</p>
        {#each $predictionFilter as filter}
          <Pill
            element={filter}
            on:remove={() => removePredictionFilter(filter)}
          />
        {/each}
      </div>
    {/if}
  </div>
  <div class="ml-auto">
    <IconButton
      filled={true}
      textColor="text-text-dark"
      on:click={() => {
        labelFilter.set([]);
        predictionFilter.set([]);
      }}
    >
      <p slot="text">Clear All</p>
      <Fa icon={faTimes} slot="icon" />
    </IconButton>
  </div>
</div>
