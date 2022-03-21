<script lang="ts">
  import SubHeading from "./SubHeading.svelte";

  export let title: string;
</script>

<div class="flex items-end border-b border-grey mb-2">
  <SubHeading heading={title} />
</div>
