<script lang="ts">
  import Icon from "svelte-fa";
  import { faTimes } from "@fortawesome/free-solid-svg-icons/faTimes";
  import { createEventDispatcher } from "svelte";

  export let element: string;

  const dispatch = createEventDispatcher();
</script>

<div
  class="flex items-center m-1 px-2 h-8 w-max rounded-full bg-highlight text-text-dark"
>
  <div class="flex h-full pr-0.5 pl-0.5 items-center">
    <p>{element}</p>
  </div>
  <div class="flex flex-auto pl-1 text-s">
    <button on:click={() => dispatch("remove")}>
      <Icon icon={faTimes} />
    </button>
  </div>
</div>
