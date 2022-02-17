export interface PatternForSample {
  patternId: number;
  probability: number;
}

export interface TooltipSpec {
  hover: boolean;
  mousePos: { x: number; y: number };
  index?: number;
  layer?: number;
}
