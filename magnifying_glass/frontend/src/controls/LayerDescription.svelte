<script lang="ts">
  import { layerWidth } from "../stores";

  export let currentLayer: string;
</script>

<div class="flex justify-center shrink-0" style={`width: ${$layerWidth}px;`}>
  <p class="truncate">
    {currentLayer}
  </p>
</div>
