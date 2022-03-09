<script lang="ts">
  import type { Writable } from "svelte/store";

  export let checked: Writable<boolean>;
</script>

<label class="relative inline-block w-10 h-6">
  <input
    class="w-0 h-0 opacity-0"
    type="checkbox"
    value={$checked}
    on:change={() => checked.set(!$checked)}
  />
  <span
    class="absolute cursor-pointer top-0 left-0 bottom-0 right-0 bg-grey transition rounded-full"
    class:bg-highlight={$checked}
  />
  <span
    class:left-0={!$checked}
    class:left-4={$checked}
    class="absolute w-5 h-5 bg-white transition rounded-full shadow m-0.5"
  />
</label>
