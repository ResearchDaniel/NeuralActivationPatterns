<script lang="ts">
  export let filled = false;
  export let plain = false;
  export let textColor = 'text-text-light';

  $: style = `${
    plain
      ? textColor
      : filled
      ? 'bg-button hover:bg-button_hover text-text-dark'
      : 'bg-white text-button border border-button hover:bg-button hover:text-white'
  } rounded-md py-1 px-2 transition`;
</script>

<button class={style} on:click on:mouseleave>
  <div class="flex items-center">
    {#if $$slots.icon}
      <div class={$$slots.text ? 'pr-2 flex' : 'flex'}>
        <slot name="icon" />
      </div>
    {/if}
    <slot name="text" />
  </div>
</button>
