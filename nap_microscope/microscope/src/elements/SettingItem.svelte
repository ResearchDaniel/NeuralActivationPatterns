<script lang="ts">
  export let title: string;
  export let indent: boolean = false;
</script>

<div
  class="flex items-center mb-2"
  class:pl-4={indent}
  class:text-text-dimmed={indent}
>
  <p>{title}</p>
  <div class="ml-auto">
    <slot />
  </div>
</div>
