<script lang="ts">
  import Pattern from "./Pattern.svelte";

  import { pinnedPatterns, pinnedPatternUids } from "../stores";
</script>

<div class="flex min-w-0 overflow-x-auto">
  {#each $pinnedPatternUids as uid}
    <Pattern samples={$pinnedPatterns[uid]} expanded={true} />
  {/each}
</div>
