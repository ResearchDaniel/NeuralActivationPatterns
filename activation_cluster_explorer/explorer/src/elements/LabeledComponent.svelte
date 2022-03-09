<script lang="ts"
>
  export let name: string;
</script>

<div class="flex items-center pr-4">
  <p class="pr-2">{name}:</p>
  <slot></slot>
</div>