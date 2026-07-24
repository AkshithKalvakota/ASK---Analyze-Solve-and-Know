import axios from 'axios'

export const api = axios.create({
  baseURL: 'http://127.0.0.1:8000',
})

// This gets set once, from a component that has access to Clerk's useAuth()
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