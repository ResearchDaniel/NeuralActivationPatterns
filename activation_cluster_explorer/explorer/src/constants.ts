import type { Config as VgConfig } from "vega";
import type { Config as VlConfig } from "vega-lite";

type Config = VgConfig | VlConfig;

export const minPatternSize = 1;
export const showAvg = false;
export const centers = 1;
export const outliers = 3;
const markColor = "#0071e3";
export const themeConfig: Config = {
  arc: { fill: markColor },
  area: { fill: markColor },
  line: { stroke: markColor },
  path: { stroke: markColor },
  rect: { fill: markColor, cornerRadius: 2 },
  bar: { fill: markColor, cornerRadiusEnd: 2 },
  circle: { fill: markColor },
  shape: { stroke: markColor },
  symbol: { fill: markColor },
  axis: {
    grid: false,
    labelColor: "#7F7F7F",
    tickColor: "#7F7F7F",
    titleFontWeight: "normal",
  },
  range: {
    category: [
      "#000000",
      "#7F7F7F",
      "#1A1A1A",
      "#999999",
      "#333333",
      "#B0B0B0",
      "#4D4D4D",
      "#C9C9C9",
      "#666666",
      "#DCDCDC",
    ],
  },
};
