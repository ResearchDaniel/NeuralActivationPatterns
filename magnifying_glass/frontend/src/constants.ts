import type { Config as VgConfig } from "vega";
import type { Config as VlConfig } from "vega-lite";

type Config = VgConfig | VlConfig;

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

export const tableau20 = [
  "#4c78a8",
  "#9ecae9",
  "#f58518",
  "#ffbf79",
  "#54a24b",
  "#88d27a",
  "#b79a20",
  "#f2cf5b",
  "#439894",
  "#83bcb6",
  "#e45756",
  "#ff9d98",
  "#79706e",
  "#bab0ac",
  "#d67195",
  "#fcbfd2",
  "#b279a2",
  "#d6a5c9",
  "#9e765f",
  "#d8b5a5",
];
