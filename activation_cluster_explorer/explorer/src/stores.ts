import { derived, writable } from "svelte/store";
import type { PatternForSample, TooltipSpec } from "./types";

export const tooltip = writable<TooltipSpec>({
  hover: false,
  mousePos: { x: 0, y: 0 },
});
export const numCenters = writable<number>(1);
export const numOutliers = writable<number>(3);
export const selectedPage = writable<string>("Overview");
export const pinnedPatterns = writable<Record<string, PatternForSample[]>>({});

export const pinnedPatternUids = derived(pinnedPatterns, ($pinnedPatterns) =>
  Object.keys($pinnedPatterns)
);
export const pages = derived(pinnedPatternUids, ($pinnedPatternUids) =>
  $pinnedPatternUids.length === 0 ? ["Overview"] : ["Overview", "Compare"]
);
