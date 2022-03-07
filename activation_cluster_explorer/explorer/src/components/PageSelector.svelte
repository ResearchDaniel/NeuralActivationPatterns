<script lang="ts">
  import { selectedPage } from "../stores";

  export let selected: boolean;
  export let value: string;
</script>

<button
  class="bg-transparent px-2 {selected
    ? 'text-text-dark'
    : 'text-text-dimmed hover:text-text-dark'}"
  {value}
  on:click={() => selectedPage.set(value)}
>
  {value}
</button>
