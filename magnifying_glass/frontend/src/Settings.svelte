<script lang="ts">
  import IconButton from "./elements/IconButton.svelte";
  import SubHeading from "./elements/SubHeading.svelte";
  import SubSubHeading from "./elements/SubSubHeading.svelte";
  import Switch from "./elements/SwitchStore.svelte";
  import NumericFieldStore from "./elements/NumericFieldStore.svelte";
  import SettingItem from "./elements/SettingItem.svelte";
  import SettingHeading from "./elements/SettingHeading.svelte";

  import Fa from "svelte-fa";
  import { faTimes } from "@fortawesome/free-solid-svg-icons/faTimes";

  import {
    compactPatterns,
    numCenters,
    numOutliers,
    removeZerosStatistics,
    showOverviewStatistics,
    settingsOpen,
    showAverage,
    showDistribution,
    showLabels,
    showPredictions,
    showProbability,
    showStatistics,
    showMaxActivating,
  } from "./stores";
</script>

{#if $settingsOpen}
  <div
    class="fixed z-10 flex top-0 bottom-0 left-0 right-0 justify-center items-center"
  >
    <div class="w-96 max-h-1/2 bg-white rounded flex flex-col shadow-2xl">
      <div class="flex items-end bg-black text-text-dark pl-2 pb-2 pr-2">
        <SubHeading heading={"Settings"} />
        <div class="ml-auto">
          <IconButton
            on:click={() => settingsOpen.set(false)}
            textColor="text-text-dark"
          >
            <Fa icon={faTimes} slot="icon" />
          </IconButton>
        </div>
      </div>
      <div class="flex flex-col overflow-y-auto p-2">
        <SettingHeading title={"General"} />
        <SettingItem title={"Show Distribution"}>
          <Switch checked={showDistribution} />
        </SettingItem>
        <SettingItem title={"Compact Patterns"}>
          <Switch checked={compactPatterns} />
        </SettingItem>
        <SettingHeading title={"Pattern Overview"} />
        <SubSubHeading heading={"Images"} />
        <SettingItem title={"Average"}>
          <Switch checked={showAverage} />
        </SettingItem>
        <SettingItem title={"Most Stable"}>
          <NumericFieldStore value={numCenters} />
        </SettingItem>
        <SettingItem title={"Least Stable"}>
          <NumericFieldStore value={numOutliers} />
        </SettingItem>
        <SubSubHeading heading={"Charts"} />
        <SettingItem title={"Statistics"}>
          <Switch checked={showStatistics} />
        </SettingItem>
        <SettingItem title={"Remove Zero Activations"} indent={true}>
          <Switch checked={removeZerosStatistics} />
        </SettingItem>
        <SettingItem title={"Statistics in Overview"} indent={true}>
          <Switch checked={showOverviewStatistics} />
        </SettingItem>
        <SettingItem title={"Probability"}>
          <Switch checked={showProbability} />
        </SettingItem>
        <SettingItem title={"Labels"}>
          <Switch checked={showLabels} />
        </SettingItem>
        <SettingItem title={"Predictions"}>
          <Switch checked={showPredictions} />
        </SettingItem>
        <SettingHeading title="Additional Explainability" />
        <SettingItem title={"Show Max Activating"}>
          <Switch checked={showMaxActivating} />
        </SettingItem>
      </div>
    </div>
  </div>
{/if}
