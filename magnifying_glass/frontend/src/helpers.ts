import type { Pattern, PatternForSample } from "./types";

import {
  numCenters,
  numOutliers,
  showAverage,
  showDistribution,
  showLabels,
  showPredictions,
  showProbability,
  showStatistics,
  showOverviewStatistics,
  removeZerosStatistics,
  compactPatterns,
  model,
  layer,
  imageFilter,
} from "./stores";

export function filterPattern(
  pattern: Pattern,
  labelFilter: string[],
  predictionFilter: string[]
): Pattern {
  return {
    ...pattern,
    samples: pattern.samples.filter((pattern) => {
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
      return true;
    }),
  };
}

export function filterPatterns(
  patterns: PatternForSample[],
  labelFilter: string[],
  predictionFilter: string[]
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
    return true;
  });
}

export function setupURLParams(urlParams: URLSearchParams) {
  if (urlParams.has("model")) model.set(urlParams.get("model"));
  if (urlParams.has("layer")) layer.set(urlParams.get("layer"));
  if (urlParams.has("showDistribution"))
    showDistribution.set(
      JSON.parse(urlParams.get("showDistribution")) as boolean
    );
  if (urlParams.has("compactPatterns"))
    compactPatterns.set(
      JSON.parse(urlParams.get("compactPatterns")) as boolean
    );
  if (urlParams.has("showAverage"))
    showAverage.set(JSON.parse(urlParams.get("showAverage")) as boolean);
  if (urlParams.has("showStatistics"))
    showStatistics.set(JSON.parse(urlParams.get("showStatistics")) as boolean);
  if (urlParams.has("removeZerosStatistics"))
    removeZerosStatistics.set(
      JSON.parse(urlParams.get("removeZerosStatistics")) as boolean
    );
  if (urlParams.has("showOverviewStatistics"))
    showOverviewStatistics.set(
      JSON.parse(urlParams.get("showOverviewStatistics")) as boolean
    );
  if (urlParams.has("showProbability"))
    showProbability.set(
      JSON.parse(urlParams.get("showProbability")) as boolean
    );
  if (urlParams.has("showLabels"))
    showLabels.set(JSON.parse(urlParams.get("showLabels")) as boolean);
  if (urlParams.has("showPredictions"))
    showPredictions.set(
      JSON.parse(urlParams.get("showPredictions")) as boolean
    );
  if (urlParams.has("numCenters"))
    numCenters.set(JSON.parse(urlParams.get("numCenters")) as number);
  if (urlParams.has("numOutliers"))
    numOutliers.set(JSON.parse(urlParams.get("numOutliers")) as number);
  if (urlParams.has("selectedImages"))
    imageFilter.set(
      JSON.parse(urlParams.get("selectedImages")) as {
        image: string;
        model: string;
      }[]
    );

  model.subscribe((setting) =>
    updateURLParams("model", `${setting}`, urlParams)
  );
  layer.subscribe((setting) =>
    updateURLParams("layer", `${setting}`, urlParams)
  );
  showDistribution.subscribe((setting) =>
    updateURLParams("showDistribution", `${setting}`, urlParams)
  );
  compactPatterns.subscribe((setting) =>
    updateURLParams("compactPatterns", `${setting}`, urlParams)
  );
  showAverage.subscribe((setting) =>
    updateURLParams("showAverage", `${setting}`, urlParams)
  );
  numCenters.subscribe((setting) =>
    updateURLParams("numCenters", `${setting}`, urlParams)
  );
  numOutliers.subscribe((setting) =>
    updateURLParams("numOutliers", `${setting}`, urlParams)
  );
  showStatistics.subscribe((setting) =>
    updateURLParams("showStatistics", `${setting}`, urlParams)
  );
  showOverviewStatistics.subscribe((setting) =>
    updateURLParams("showOverviewStatistics", `${setting}`, urlParams)
  );
  removeZerosStatistics.subscribe((setting) =>
    updateURLParams("removeZerosStatistics", `${setting}`, urlParams)
  );
  showProbability.subscribe((setting) =>
    updateURLParams("showProbability", `${setting}`, urlParams)
  );
  showLabels.subscribe((setting) =>
    updateURLParams("showLabels", `${setting}`, urlParams)
  );
  showPredictions.subscribe((setting) =>
    updateURLParams("showPredictions", `${setting}`, urlParams)
  );
  imageFilter.subscribe((setting) => {
    updateURLParams("selectedImages", JSON.stringify(setting), urlParams);
  });
}

function updateURLParams(
  settingName: string,
  setting: string,
  urlParams: URLSearchParams
) {
  urlParams.set(settingName, setting);
  window.history.replaceState({}, "", `${location.pathname}?${urlParams}`);
}
