<script lang="ts">
  export let filled = false;
  export let fullHeight = false;
  export let textColor = "text-text-light";
</script>

<button
  class={`${textColor} rounded-md py-1 px-2 transition`}
  class:bg-button={filled}
  class:h-full={fullHeight}
  on:click
>
  <div class="flex items-center ">
    {#if $$slots.icon}
      <div class={$$slots.text ? "pr-2 flex" : "flex"}>
        <slot name="icon" />
      </div>
    {/if}
    <slot name="text" />
  </div>
</button>
