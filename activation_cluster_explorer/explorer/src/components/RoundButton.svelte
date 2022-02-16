<script lang="ts">
  export let filled = false;
  export let large = false;

  $: style = `${
    filled
      ? 'bg-button hover:bg-button_hover text-text-dark'
      : 'bg-white text-button border border-button hover:bg-button hover:text-white'
  } rounded-full transition ${large ? 'text-2xl h-10 w-10' : 'h-6 w-6'}`;
</script>

<button class={style} on:click on:mouseleave>
  <div class="flex items-center justify-center">
    {#if $$slots.icon}
      <div class="flex">
        <slot name="icon" />
      </div>
    {/if}
  </div>
</button>
