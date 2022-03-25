export interface PatternForSample {
  patternUid: string;
  patternId: number;
  probability: number;
  outlierScore: number;
  fileName: string;
  model: string;
  layer: string;
  filter?: string;
  label?: string;
  prediction?: string;
}

export interface TooltipSpec {
  hover: boolean;
  mousePos: { x: number; y: number };
  sample?: PatternForSample;
  layer?: string;
}

export interface Patterns {
  samples: PatternForSample[];
  persistence: number[];
  statistics?: Statistics[];
}

export interface Pattern {
  samples: PatternForSample[];
  persistence: number;
  statistics: Statistics;
}

export interface Statistics {
  IQR: number[];
  lower: number[];
  max: number[];
  mean: number[];
  min: number[];
  q1: number[];
  q3: number[];
}
