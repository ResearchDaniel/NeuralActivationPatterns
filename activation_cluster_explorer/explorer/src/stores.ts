import { derived, writable } from "svelte/store";
import { minPatternSize } from "./constants";
import type { PatternForSample, TooltipSpec } from "./types";

export const tooltip = writable<TooltipSpec>({
  hover: false,
  mousePos: { x: 0, y: 0 },
});
export const numCenters = writable<number>(1);
export const numOutliers = writable<number>(3);
export const selectedPage = writable<string>("Overview");
export const pinnedPatterns = writable<Record<string, PatternForSample[]>>({});
export const patternFilter = writable<{ label: string; patternId: number }[]>(
  []
);
export const labelFilter = writable<string[]>([]);
export const predictionFilter = writable<string[]>([]);

export const filteredPinnedPatterns = derived(
  [pinnedPatterns, labelFilter, predictionFilter],
  ([$pinnedPatterns, $labelFilter, $predictionFilter]) => {
    const filtered = {};
    const pinnedPatternUids = Object.keys($pinnedPatterns);
    pinnedPatternUids.forEach((uid) => {
      const patterns = $pinnedPatterns[uid].filter((pattern) => {
        if ($labelFilter.length !== 0) {
          if (!$labelFilter.includes(`${pattern.label}`)) {
            return false;
          }
        }
        if ($predictionFilter.length !== 0) {
          if (!$predictionFilter.includes(`${pattern.prediction}`)) {
            return false;
          }
        }
        return true;
      });
      if (patterns.length >= minPatternSize) filtered[uid] = patterns;
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
