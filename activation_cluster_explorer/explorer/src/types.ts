export interface PatternForSample {
  patternId: number;
  probability: number;
  outlier_score: number;
  file_name: string;
}

export interface TooltipSpec {
  hover: boolean;
  mousePos: { x: number; y: number };
  sample?: PatternForSample;
  layer?: number;
}
