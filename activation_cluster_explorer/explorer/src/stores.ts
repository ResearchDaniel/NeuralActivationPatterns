import { writable } from "svelte/store";
import type { TooltipSpec } from "./types";

export const tooltip = writable<TooltipSpec>({
  hover: false,
  mousePos: { x: 0, y: 0 },
});
export const numCenters = writable<number>(1);
export const numOutliers = writable<number>(3);
