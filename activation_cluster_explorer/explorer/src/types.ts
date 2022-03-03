export interface PatternForSample {
  patternId: number;
  probability: number;
  outlierScore: number;
  fileName: string;
  label?: string;
  prediction?: string;
}

export interface TooltipSpec {
  hover: boolean;
  mousePos: { x: number; y: number };
  sample?: PatternForSample;
  layer?: number;
}
