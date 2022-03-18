<script lang="ts">
  import PageSelector from "./elements/PageSelector.svelte";
  import IconButton from "./elements/IconButton.svelte";

  import Fa from "svelte-fa";
  import { faGithub } from "@fortawesome/free-brands-svg-icons/faGithub";
  import { faCog } from "@fortawesome/free-solid-svg-icons/faCog";

  import { pages, selectedPage, settingsOpen } from "./stores";
</script>

<nav class="p-2 pl-3 bg-black flex">
  <div class="flex items-center">
    <h2 class="text-2xl font-bold text-text-dark">NAP Microscope</h2>
    <div class="flex pl-12 overflow-auto">
      {#each $pages as page}
        <PageSelector selected={page === $selectedPage} value={page} />
      {/each}
    </div>
  </div>
  <div class="flex items-center pl-4 ml-auto">
    <IconButton
      on:click={() => settingsOpen.set(true)}
      textColor="text-text-dark"
    >
      <Fa icon={faCog} slot="icon" />
    </IconButton>
  </div>
  <div class="flex items-center pl-4">
    <IconButton
      on:click={() =>
        window.open(
          "https://github.com/ResearchDaniel/NetworkActivationPatterns",
          "_blank"
        )}
      textColor="text-text-dark"
    >
      <Fa icon={faGithub} slot="icon" />
    </IconButton>
  </div>
</nav>
