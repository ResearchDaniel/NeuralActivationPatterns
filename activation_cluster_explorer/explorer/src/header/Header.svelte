<script lang="ts">
  import Promo from "./Promo.svelte";
  import PageSelector from "../components/PageSelector.svelte";

  import { pages, selectedPage } from "../stores";
</script>

<nav class="p-2 pl-3 bg-black flex justify-between">
  <div class="flex items-center">
    <h2 class="text-2xl font-bold text-text-dark">Activation Clusters</h2>
    <div class="flex pl-12 overflow-auto">
      {#each $pages as page}
        <PageSelector selected={page === $selectedPage} value={page} />
      {/each}
    </div>
  </div>
  <div class="flex items-center pl-4">
    <Promo />
  </div>
</nav>
