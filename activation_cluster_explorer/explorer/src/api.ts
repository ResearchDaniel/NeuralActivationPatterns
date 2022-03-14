import type { PatternForSample, Patterns } from "./types";

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

export async function fetchDataset(model: string): Promise<
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

export async function fetchLabels(model: string) {
  return fetch(`/api/get_labels/${model}`).then((response) => response.json());
}

export async function fetchFilterMethods(
  model: string,
  layer: string
): Promise<string[]> {
  return fetch(`/api/get_filter_methods/${model}/${layer}`)
    .then((response) => response.json())
    .then((jsonResponse) => [...(jsonResponse["methods"] as string[]).sort()]);
}

export async function fetchFilters(
  model: string,
  layer: string,
  filterMethod: string
): Promise<string[]> {
  return fetch(`/api/get_filters/${model}/${layer}/${filterMethod}`)
    .then((response) => response.json())
    .then((jsonResponse) => [
      "---",
      ...(jsonResponse["filters"] as string[]).sort((a, b) => +a - +b),
    ]);
}

export async function fetchPatterns(
  model: string,
  layer: string,
  filter?: string,
  filterMethod?: string
): Promise<Patterns> {
  if (model === undefined || layer === undefined)
    return { samples: [], persistence: [] };
  const dataset = await fetchDataset(model);
  const labels = await fetchLabels(model);
  if (dataset.length === 0 || labels === undefined)
    return { samples: [], persistence: [] };
  const infoResponse =
    filter === "---" || filterMethod === undefined
      ? await fetch(`/api/get_pattern_info/${model}/${layer}`)
      : await fetch(
          `/api/get_filter_pattern_info/${model}/${layer}/${filterMethod}/${filter}`
        );
  if (!infoResponse.ok) return { samples: [], persistence: [] };
  const infoJsonResponse = await infoResponse.json();
  const info = JSON.parse(infoJsonResponse);
  const response =
    filter === "---"
      ? await fetch(`/api/get_patterns/${model}/${layer}`)
      : await fetch(
          `/api/get_filter_patterns/${model}/${layer}/${filterMethod}/${filter}`
        );
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
            filter: filter === "---" ? undefined : filter,
            filterMethod: filter === "---" ? undefined : filterMethod,
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
