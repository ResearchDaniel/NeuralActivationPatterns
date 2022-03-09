<script lang="ts">
  import type { Writable } from "svelte/store";

  export let value: Writable<number>;
  let val = $value;

  $: value.set(val);
</script>

<input
  class="rounded border w-32 h-10 border-grey"
  type="number"
  bind:value={val}
/>
