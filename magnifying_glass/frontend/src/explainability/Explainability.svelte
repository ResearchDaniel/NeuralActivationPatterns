<script lang="ts">
  import MaxActivations from "./MaxActivations.svelte";

  export let maxActivating: string[];
</script>

<div class="flex flex-1 flex-col min-h-0 h-full">
  <MaxActivations {maxActivating} />
</div>
