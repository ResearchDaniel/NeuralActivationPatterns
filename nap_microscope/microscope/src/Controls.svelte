<script lang="ts">
  import Select from "svelte-select";

  import { fetchLayers, fetchModels, fetchPatterns } from "./api";
  import type { Patterns } from "./types";

  export let patternsRequest: Promise<Patterns> = undefined;

  let model: string = undefined;
  let layer: string = undefined;

  $: patternsRequest = fetchPatterns(model, layer);
</script>

<div class="flex flex-col w-full">
  {#await fetchModels() then models}
    <div class="pt-2">
      <Select
        placeholder="Model"
        items={models}
        on:select={(event) => {
          model = event.detail.value;
          layer = undefined;
        }}
        on:clear={() => {
          model = undefined;
          layer = undefined;
        }}
      />
    </div>
  {/await}
  {#if model !== undefined}
    {#await fetchLayers(model) then layers}
      <div class="pt-2">
        <Select
          placeholder="Layer"
          items={layers}
          on:select={(event) => {
            layer = event.detail.value;
          }}
          on:clear={() => {
            layer = undefined;
          }}
        />
      </div>
    {/await}
  {/if}
</div>
