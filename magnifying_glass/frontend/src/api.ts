import { fromArrow } from "arquero";

import type ColumnTable from "arquero/dist/types/table/column-table";
import type { PatternForSample, Patterns, Pattern } from "./types";

export async function fetchModels(): Promise<string[]> {
  const response = await fetch(`/api/get_models`);
  const jsonResponse = await response.json();
  const models = jsonResponse["networks"] as string[];
  return models;
}

export async function fetchLayers(model: string): Promise<string[]> {
  return fetch(`/api/get_layers/${model}`)
    .then((response) => response.json())
    .then((jsonResponse) => jsonResponse["layers"] as string[]);
}

async function fetchDataset(model: string): Promise<
  {
    file_name: string;
    label?: string;
    prediction?: string;
  }[]
> {
  return fetch(`/api/get_dataset/${model}`)
    .then((response) => response.json())
    .then((jsonResponse) => JSON.parse(jsonResponse));
}

async function fetchLabels(model: string) {
  return fetch(`/api/get_labels/${model}`).then((response) => response.json());
}

export async function fetchPatternStatistics(
  model: string,
  layer: string,
  pattern: number,
  key: string
): Promise<{ table: ColumnTable; key: string } | undefined> {
  return fetch(`/api/get_pattern_statistics/${model}/${layer}/${pattern}`).then(
    (response) =>
      response.ok
        ? response.arrayBuffer().then((buffer) => {
            return { table: fromArrow(new DataView(buffer)), key: key };
          })
        : undefined
  );
}

export async function fetchPatterns(
  model: string,
  layer: string
): Promise<Patterns> {
  if (model === undefined || layer === undefined)
    return { samples: [], persistence: [] };
  const dataset = await fetchDataset(model);
  const labels = await fetchLabels(model);
  if (dataset.length === 0 || labels === undefined)
    return { samples: [], persistence: [] };
  const infoResponse = await fetch(`/api/get_pattern_info/${model}/${layer}`);
  if (!infoResponse.ok) return { samples: [], persistence: [] };
  const infoJsonResponse = await infoResponse.json();
  const info = JSON.parse(infoJsonResponse);
  const response = await fetch(`/api/get_patterns/${model}/${layer}`);
  if (!response.ok) return { samples: [], persistence: [] };
  const jsonResponse = await response.json();
  const patterns = JSON.parse(jsonResponse);
  if (patterns.length !== dataset.length)
    return { samples: [], persistence: [] };
  return {
    samples: patterns
      .map(
        (
          pattern: {
            patternId: number;
            probability: number;
            outlier_score: number;
          },
          index: number
        ) => {
          return {
            patternUid: `${model}_${layer}_${pattern.patternId}`,
            model: model,
            layer: layer,
            patternId: pattern.patternId,
            probability: pattern.probability,
            outlierScore: pattern.outlier_score,
            fileName: dataset[index].file_name,
            labelIndex: dataset[index].label,
            label: labels[dataset[index].label],
            predictionIndex: dataset[index].prediction,
            prediction: labels[dataset[index].prediction],
          } as PatternForSample;
        }
      )
      .filter((pattern) => pattern.patternId >= 0),
    persistence: info.map((infoElement) => infoElement.pattern_persistence),
  } as Patterns;
}

export async function fetchMaxActivating(
  model: string,
  layer: string,
  showMaxActivating: boolean
): Promise<string[]> {
  if (!showMaxActivating) return [];
  if (model === undefined || layer === undefined) return [];
  return fetch(`/api/get_max_activations/${model}/${layer}`)
    .then((response) => response.json())
    .then((jsonResponse) => jsonResponse["max_activations"] as string[]);
}

export async function fetchFeatureVisExists(
  model: string,
  layer: string
): Promise<boolean> {
  return fetch(`/api/get_layer_feature_vis_exists/${model}/${layer}`).then(
    (response) => response.json()
  );
}

export async function fetchPatternsForImages(
  images: {
    image: string;
    model: string;
  }[]
): Promise<Pattern[]> {
  const response = await fetch("/api/get_image_patterns", {
    method: "POST",
    body: JSON.stringify(images),
  });
  const jsonResponse = await response.json();
  return jsonResponse.map((patternResponse) => {
    const samples = JSON.parse(patternResponse["samples"]);
    const statistics =
      patternResponse["statistics"] === undefined
        ? undefined
        : JSON.parse(patternResponse["statistics"]);
    const persistence = patternResponse["persistence"];
    const model = patternResponse["model"];
    const layer = patternResponse["layer"];
    const labels = patternResponse["labels"];
    return {
      samples: samples.map((sample) => {
        return {
          patternUid: `${model}_${layer}_${sample.patternId}`,
          model: model,
          layer: layer,
          patternId: sample.patternId,
          probability: sample.probability,
          outlierScore: sample.outlier_score,
          fileName: sample.file_name,
          labelIndex: sample.label,
          label: labels[sample.label],
          predictionIndex: sample.prediction,
          prediction: labels[sample.prediction],
        } as PatternForSample;
      }),
      persistence: persistence,
      statistics: statistics,
    } as Pattern;
  }) as Pattern[];
}
