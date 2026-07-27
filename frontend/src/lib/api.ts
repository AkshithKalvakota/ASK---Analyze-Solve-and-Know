import axios from 'axios'

export const api = axios.create({
  baseURL: 'http://127.0.0.1:8000',
})

let getTokenFn: (() => Promise<string | null>) | null = null

export function setTokenGetter(fn: () => Promise<string | null>) {
  getTokenFn = fn
}

api.interceptors.request.use(async (config) => {
  if (getTokenFn) {
    const token = await getTokenFn()
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
  }
  return config
})

export interface Project {
  id: string
  name: string
  created_at: string
}

export async function fetchProjects(): Promise<Project[]> {
  const res = await api.get('/projects')
  return res.data
}

export async function createProject(name: string): Promise<Project> {
  const res = await api.post('/projects', { name })
  return res.data
}

export async function deleteProject(id: string): Promise<void> {
  await api.delete(`/projects/${id}`)
}

export interface Dataset {
  id: string
  project_id: string
  filename: string
  content_type: string
  profile_result?: {
    dtypes: Record<string, string>
    quality_score: number
    n_rows: number
    n_columns: number
    [key: string]: any
  }
  target_column?: string | null
  problem_type?: string | null
  created_at: string
}

export async function uploadDataset(projectId: string, file: File): Promise<Dataset> {
  const formData = new FormData()
  formData.append('file', file)
  const res = await api.post(`/projects/${projectId}/datasets`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return res.data
}

export async function fetchDatasets(projectId: string): Promise<Dataset[]> {
  const res = await api.get(`/projects/${projectId}/datasets`)
  return res.data
}

export async function profileDataset(projectId: string, datasetId: string): Promise<Dataset> {
  const res = await api.post(`/projects/${projectId}/datasets/${datasetId}/profile`)
  return res.data
}

export async function setTargetColumn(
  projectId: string,
  datasetId: string,
  targetColumn: string,
  problemTypeOverride?: string
): Promise<Dataset> {
  const res = await api.post(`/projects/${projectId}/datasets/${datasetId}/target`, {
    target_column: targetColumn,
    problem_type_override: problemTypeOverride ?? null,
  })
  return res.data
}

export interface TrainedModel {
  id: string
  dataset_id: string
  model_name: string
  metrics: Record<string, number>
  all_results: Record<string, Record<string, number>>
  created_at: string
}

export async function trainModel(projectId: string, datasetId: string): Promise<TrainedModel> {
  const res = await api.post(`/projects/${projectId}/datasets/${datasetId}/models`)
  return res.data
}

export async function fetchModels(projectId: string, datasetId: string): Promise<TrainedModel[]> {
  const res = await api.get(`/projects/${projectId}/datasets/${datasetId}/models`)
  return res.data
}

export interface InputField {
  name: string
  type: string
}

export async function fetchInputSchema(
  projectId: string,
  datasetId: string,
  modelId: string
): Promise<{ fields: InputField[] }> {
  const res = await api.get(
    `/projects/${projectId}/datasets/${datasetId}/models/${modelId}/input-schema`
  )
  return res.data
}

export async function predict(
  projectId: string,
  datasetId: string,
  modelId: string,
  values: Record<string, any>
): Promise<{ prediction: string | number }> {
  const res = await api.post(
    `/projects/${projectId}/datasets/${datasetId}/models/${modelId}/predict`,
    { values }
  )
  return res.data
}

export interface FeatureImportance {
  feature_importance: { feature: string; importance: number }[]
}

export interface PredictionExplanation {
  base_value: number
  contributions: { feature: string; impact: number }[]
  plain_english: string[]
  sum_check: number
}

export async function fetchFeatureImportance(
  projectId: string,
  datasetId: string,
  modelId: string
): Promise<FeatureImportance> {
  const res = await api.get(
    `/projects/${projectId}/datasets/${datasetId}/models/${modelId}/feature-importance`
  )
  return res.data
}

export async function explainPrediction(
  projectId: string,
  datasetId: string,
  modelId: string,
  values: Record<string, any>
): Promise<PredictionExplanation> {
  const res = await api.post(
    `/projects/${projectId}/datasets/${datasetId}/models/${modelId}/explain`,
    { values }
  )
  return res.data
}