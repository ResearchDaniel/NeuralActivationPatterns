<script lang="ts">
  import type { Writable } from "svelte/store";

  export let value: Writable<number>;
  export let min: number = 0;

  let val = $value;

  $: value.set(val);
</script>

<input
  class="rounded border h-8 w-32 border-grey"
  type="number"
  bind:value={val}
  {min}
/>
