<script lang="ts">
  import Pattern from "./Pattern.svelte";

  import {
    pinnedPatterns,
    filteredPinnedPatternUids,
    filteredPinnedPatterns,
  } from "../stores";
</script>

<div class="flex min-w-0 overflow-x-auto">
  {#each $filteredPinnedPatternUids as uid}
    <Pattern
      samples={$pinnedPatterns[uid]}
      expanded={true}
      filteredSamples={$filteredPinnedPatterns[uid]}
    />
  {/each}
</div>
