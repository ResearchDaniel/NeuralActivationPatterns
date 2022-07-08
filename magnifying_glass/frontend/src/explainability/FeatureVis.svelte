<script lang="ts">
  import SubHeading from "../elements/SubHeading.svelte";
  import { model, layer } from "../stores";
</script>

<div class="flex flex-col pb-2">
  <SubHeading heading={`Feature Vis`} />
  <div class="relative p-1 h-32">
    <img
      class="h-32"
      src={`/api/get_layer_feature_vis/${$model}/${$layer}`}
      alt="Layer Level Feature Visualization"
    />
  </div>
</div>
