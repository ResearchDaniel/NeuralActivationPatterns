<script lang="ts">
  import IconButton from "./elements/IconButton.svelte";
  import SubHeading from "./elements/SubHeading.svelte";
  import Switch from "./elements/SwitchStore.svelte";

  import Fa from "svelte-fa";
  import { faTimes } from "@fortawesome/free-solid-svg-icons/faTimes";

  import {
    numCenters,
    numOutliers,
    settingsOpen,
    showAverage,
    showDistribution,
  } from "./stores";
  import NumericFieldStore from "./elements/NumericFieldStore.svelte";
</script>

{#if $settingsOpen}
  <div
    class="fixed z-10 flex top-0 bottom-0 left-0 right-0 justify-center items-center"
  >
    <div class="w-96 max-h-1/2 bg-white rounded flex flex-col p-2 shadow-2xl">
      <div class="flex items-end border-b-2 border-grey">
        <SubHeading heading={"Settings"} />
        <div class="ml-auto">
          <IconButton on:click={() => settingsOpen.set(false)}>
            <Fa icon={faTimes} slot="icon" />
          </IconButton>
        </div>
      </div>
      <div class="flex flex-col mt-4 overflow-y-auto">
        <div class="flex items-center mb-2">
          <p>Show Average</p>
          <div class="ml-auto">
            <Switch checked={showAverage} />
          </div>
        </div>
        <div class="flex items-center mb-2">
          <p>Show Distribution</p>
          <div class="ml-auto">
            <Switch checked={showDistribution} />
          </div>
        </div>
        <div class="flex items-center mb-2">
          <p>Center Samples</p>
          <div class="ml-auto">
            <NumericFieldStore value={numCenters} />
          </div>
        </div>
        <div class="flex items-center mb-2">
          <p>Outlier Samples</p>
          <div class="ml-auto">
            <NumericFieldStore value={numOutliers} />
          </div>
        </div>
      </div>
    </div>
  </div>
{/if}
