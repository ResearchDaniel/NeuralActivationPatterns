<script lang="ts">
  import Pattern from "./Pattern.svelte";

  import {
    pinnedPatterns,
    filteredPinnedPatternUids,
    filteredPinnedPatterns,
  } from "../stores";

  let width: number;

  $: patternWidth = Math.max(width / $filteredPinnedPatternUids.length, 800);
</script>

<div class="flex min-w-0 overflow-x-auto h-full p-2" bind:clientWidth={width}>
  {#each $filteredPinnedPatternUids as uid}
    <Pattern
      pattern={$pinnedPatterns[uid]}
      expanded={true}
      {patternWidth}
      filteredSamples={$filteredPinnedPatterns[uid].samples}
    />
  {/each}
</div>
