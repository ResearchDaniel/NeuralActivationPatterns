export interface PatternForSample {
  patternUid: string;
  patternId: number;
  probability: number;
  outlierScore: number;
  fileName: string;
  model: string;
  layer: string;
  label?: string;
  prediction?: string;
}

export interface TooltipSpec {
  hover: boolean;
  mousePos: { x: number; y: number };
  sample?: PatternForSample;
  layer?: number;
}
