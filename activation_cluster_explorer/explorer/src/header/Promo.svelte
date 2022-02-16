<script>
  import Fa from "svelte-fa";
  import { faGithub } from "@fortawesome/free-brands-svg-icons/faGithub";

  import IconButton from "../components/IconButton.svelte";
</script>

<div>
  <IconButton
    on:click={() =>
      window.open(
        "https://github.com/ResearchDaniel/NetworkActivationClustering",
        "_blank"
      )}
    plain={true}
    textColor="text-text-dark"
  >
    <Fa icon={faGithub} slot="icon" />
  </IconButton>
</div>
