import type { PatternForSample } from "./types";

export function filterPatterns(
  patterns: PatternForSample[],
  labelFilter: string[],
  predictionFilter: string[],
  imageFilter: string[]
): PatternForSample[] {
  return patterns.filter((pattern) => {
    if (labelFilter.length !== 0) {
      if (!labelFilter.includes(`${pattern.label}`)) {
        return false;
      }
    }
    if (predictionFilter.length !== 0) {
      if (!predictionFilter.includes(`${pattern.prediction}`)) {
        return false;
      }
    }
    if (imageFilter.length !== 0) {
      if (!imageFilter.includes(pattern.fileName)) {
        return false;
      }
    }
    return true;
  });
}
