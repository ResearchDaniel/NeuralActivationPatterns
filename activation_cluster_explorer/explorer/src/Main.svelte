<script lang="ts">
  import Patterns from "./patterns/Patterns.svelte";

  import type { PatternForSample } from "./types";

  export let patterns: PatternForSample[];
</script>

<div class="flex flex-col flex-grow p-2 min-h-0">
  <Patterns {patterns} />
</div>
