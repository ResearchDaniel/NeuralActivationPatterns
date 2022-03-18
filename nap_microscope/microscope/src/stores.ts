import { derived, writable } from "svelte/store";
import { filterPatterns } from "./helpers";
import type { PatternForSample, TooltipSpec } from "./types";

export const settingsOpen = writable<boolean>(false);
export const tooltip = writable<TooltipSpec>({
  hover: false,
  mousePos: { x: 0, y: 0 },
});
export const numCenters = writable<number>(1);
export const numOutliers = writable<number>(3);
export const minPatternSize = writable<number>(3);
export const showAverage = writable<boolean>(false);
export const showDistribution = writable<boolean>(false);
export const selectedPage = writable<string>("Overview");
export const pinnedPatterns = writable<Record<string, PatternForSample[]>>({});
export const patternFilter = writable<{ label: string; patternId: number }[]>(
  []
);
export const labelFilter = writable<string[]>([]);
export const predictionFilter = writable<string[]>([]);
export const imageFilter = writable<string[]>([]);

export const filteredPinnedPatterns = derived(
  [pinnedPatterns, labelFilter, predictionFilter, imageFilter, minPatternSize],
  ([
    $pinnedPatterns,
    $labelFilter,
    $predictionFilter,
    $imageFilter,
    $minPatternSize,
  ]) => {
    const filtered = {};
    const pinnedPatternUids = Object.keys($pinnedPatterns);
    pinnedPatternUids.forEach((uid) => {
      const patterns = filterPatterns(
        $pinnedPatterns[uid],
        $labelFilter,
        $predictionFilter,
        $imageFilter
      );
      if (patterns.length >= $minPatternSize) filtered[uid] = patterns;
    });
    return filtered;
  }
);
export const filteredPinnedPatternUids = derived(
  filteredPinnedPatterns,
  ($pinnedPatterns) => Object.keys($pinnedPatterns)
);
export const pages = derived(filteredPinnedPatternUids, ($pinnedPatternUids) =>
  $pinnedPatternUids.length === 0 ? ["Overview"] : ["Overview", "Compare"]
);