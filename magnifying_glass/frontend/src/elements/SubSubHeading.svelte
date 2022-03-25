<script lang="ts">
  export let heading: string;
  export let color: string = undefined;
</script>

<h2
  class="text-l font-bold leading-7 pr-2"
  style={color !== undefined ? `color: ${color}` : ""}
>
  {heading}
</h2>
