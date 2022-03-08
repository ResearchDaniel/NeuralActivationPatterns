<script lang="ts">
  export let items: string[] | number[];
  export let value: string | number;
</script>

<select bind:value on:change class="rounded border border-grey">
  {#each items as item}
    <option value={item}>
      {item}
    </option>
  {/each}
</select>
