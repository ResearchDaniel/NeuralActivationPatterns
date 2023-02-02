import { derived, writable } from "svelte/store";
import { filterPattern } from "./helpers";
import type { Pattern, TooltipSpec } from "./types";

export const model = writable<string>(undefined);
export const layer = writable<string>();
export const settingsOpen = writable<boolean>(false);
export const tooltip = writable<TooltipSpec>({
  hover: false,
  mousePos: { x: 0, y: 0 },
});
export const numCenters = writable<number>(1);
export const numOutliers = writable<number>(0);
export const minPatternSize = writable<number>(3);
export const showAverage = writable<boolean>(false);
export const showDistribution = writable<boolean>(false);
export const compactPatterns = writable<boolean>(true);
export const showProbability = writable<boolean>(false);
export const showStatistics = writable<boolean>(true);
export const showOverviewStatistics = writable<boolean>(false);
export const removeZerosStatistics = writable<boolean>(false);
export const showLabels = writable<boolean>(false);
export const showPredictions = writable<boolean>(false);
export const showMaxActivating = writable<boolean>(true);
export const selectedPage = writable<string>("Overview");
export const pinnedPatterns = writable<Record<string, Pattern>>({});
export const patternFilter = writable<{ label: string; patternId: number }[]>(
  []
);
export const labelFilter = writable<string[]>([]);
export const predictionFilter = writable<string[]>([]);
export const imageFilter = writable<{ image: string; model: string }[]>([]);
export const layerWidth = writable<number>(80);
export const layerHeight = writable<number>(80);
export const patternsWidth = writable<number>(0);

export const filteredPinnedPatterns = derived(
  [pinnedPatterns, labelFilter, predictionFilter, minPatternSize],
  ([$pinnedPatterns, $labelFilter, $predictionFilter, $minPatternSize]) => {
    const filtered = {};
    const pinnedPatternUids = Object.keys($pinnedPatterns);
    pinnedPatternUids.forEach((uid) => {
      const patterns = filterPattern(
        $pinnedPatterns[uid],
        $labelFilter,
        $predictionFilter
      );
      if (patterns.samples.length >= $minPatternSize) filtered[uid] = patterns;
    });
    return filtered;
  }
);
export const filteredPinnedPatternUids = derived(
  filteredPinnedPatterns,
  ($pinnedPatterns) => Object.keys($pinnedPatterns)
);
export const pages = derived(
  [filteredPinnedPatternUids, imageFilter],
  ([$pinnedPatternUids, $imageFilter]) => {
    let pages = ["Overview"];
    pages = $pinnedPatternUids.length === 0 ? pages : [...pages, "Compare"];
    pages = $imageFilter.length === 0 ? pages : [...pages, "Images"];
    return pages;
  }
);
