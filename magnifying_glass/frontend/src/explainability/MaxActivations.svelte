<script lang="ts">
  import SubHeading from "../elements/SubHeading.svelte";
  import { model, layer } from "../stores";

  export let maxActivating: string[];
</script>

<div class="flex flex-col min-h-0 h-full">
  <SubHeading heading={`Max Activations`} />
  <div class="flex items-start overflow-y-auto min-h-0 pt-2 flex-wrap">
    {#each maxActivating as maxImage}
      <div class="relative p-1">
        <img
          class="h-32"
          src={`/api/get_max_activation_image/${$model}/${$layer}/${maxImage}`}
          alt="Max Activation Data Sample"
        />
      </div>
    {/each}
  </div>
</div>
