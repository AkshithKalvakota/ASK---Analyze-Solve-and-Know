import { useEffect, useState } from 'react'
import { useAuth, UserButton } from '@clerk/clerk-react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  setTokenGetter,
  fetchProjects,
  createProject,
  deleteProject,
  uploadDataset,
  fetchDatasets,
  deleteDataset,
  profileDataset,
  setTargetColumn,
  trainModel,
  fetchModels,
  fetchInputSchema,
  predict,
  fetchFeatureImportance,
  explainPrediction,
  type Dataset,
  type PredictionExplanation,
} from '../lib/api'

function QualityScoreBadge({ score }: { score: number }) {
  const color =
    score >= 80 ? 'bg-[#4FD1C5] text-[#0B1220]' : score >= 50 ? 'bg-[#E8B04B] text-[#0B1220]' : 'bg-[#E85D5D] text-white'
  const label = score >= 80 ? 'Good' : score >= 50 ? 'Fair' : 'Poor'

  return (
    <div className="flex items-center gap-2 mt-3">
      <div className={`${color} font-mono text-xs font-bold rounded-full w-10 h-10 flex items-center justify-center`}>
        {Math.round(score)}
      </div>
      <p className="text-xs text-[#8A93A6]">
        Data Quality: <span className="text-[#EDEEF2]">{label}</span>
      </p>
    </div>
  )
}

function MetricsDisplay({ metrics }: { metrics: Record<string, number> }) {
  const labels: Record<string, string> = {
    r2: 'R² Score',
    mae: 'Mean Abs. Error',
    rmse: 'RMSE',
    accuracy: 'Accuracy',
    precision: 'Precision',
    recall: 'Recall',
    f1: 'F1 Score',
    roc_auc: 'ROC AUC',
  }
  const isPercentMetric = (key: string) =>
    ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'].includes(key)

  return (
    <div className="grid grid-cols-3 gap-2 mt-3">
      {Object.entries(metrics).map(([key, value]) => (
        <div key={key} className="bg-[#0B1220] border border-[#2A3550] rounded p-2 text-center">
          <p className="text-[10px] text-[#8A93A6] uppercase tracking-wide">{labels[key] ?? key}</p>
          <p className="font-mono text-sm font-semibold text-[#4FD1C5] mt-1">
            {isPercentMetric(key) ? `${(value * 100).toFixed(1)}%` : value}
          </p>
        </div>
      ))}
    </div>
  )
}

function FeatureImportanceDisplay({
  projectId,
  datasetId,
  modelId,
}: {
  projectId: string
  datasetId: string
  modelId: string
}) {
  const { data, isLoading } = useQuery({
    queryKey: ['feature-importance', modelId],
    queryFn: () => fetchFeatureImportance(projectId, datasetId, modelId),
  })

  if (isLoading) return <p className="text-xs text-[#8A93A6] mt-3">Loading feature importance...</p>
  if (!data || data.feature_importance.length === 0) return null

  const maxImportance = Math.max(...data.feature_importance.map((f) => f.importance))

  return (
    <div className="mt-4">
      <div className="flex items-center gap-2 mb-3">
        <span className="w-1.5 h-1.5 rounded-full bg-[#4FD1C5]" />
        <p className="text-xs text-[#4FD1C5] font-mono tracking-widest font-semibold">FEATURE IMPORTANCE</p>
    </div>
      <div className="space-y-2">
        {data.feature_importance.slice(0, 8).map((f) => (
          <div key={f.feature} className="flex items-center gap-3">
            <span className="text-xs text-[#EDEEF2] w-28 truncate" title={f.feature}>{f.feature}</span>
            <div className="flex-1 bg-[#0B1220] rounded h-3">
              <div
                className="bg-[#4FD1C5] h-3 rounded"
                style={{ width: `${(f.importance / maxImportance) * 100}%` }}
              />
            </div>
            <span className="font-mono text-xs text-[#8A93A6] w-14 text-right">{f.importance}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function PredictionForm({
  projectId,
  datasetId,
  modelId,
}: {
  projectId: string
  datasetId: string
  modelId: string
}) {
  const [values, setValues] = useState<Record<string, string>>({})
  const [result, setResult] = useState<string | number | null>(null)
  const [previousResult, setPreviousResult] = useState<string | number | null>(null)
  const [explanation, setExplanation] = useState<PredictionExplanation | null>(null)

  const { data: schema } = useQuery({
    queryKey: ['input-schema', modelId],
    queryFn: () => fetchInputSchema(projectId, datasetId, modelId),
  })

  const predictMutation = useMutation({
    mutationFn: () => predict(projectId, datasetId, modelId, values),
    onSuccess: async (data) => {
      setPreviousResult(result)
      setResult(data.prediction)
      const exp = await explainPrediction(projectId, datasetId, modelId, values)
      setExplanation(exp)
    },
  })

  function handleChange(fieldName: string, value: string) {
    setValues((prev) => ({ ...prev, [fieldName]: value }))
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    predictMutation.mutate()
  }

  if (!schema) return <p className="text-xs text-[#8A93A6] mt-3">Loading input fields...</p>

  const changed = previousResult !== null && result !== null && previousResult !== result

  return (
    <form onSubmit={handleSubmit} className="mt-4 bg-[#0B1220] border border-[#2A3550] rounded-lg p-4">
      <div className="flex items-center gap-2 mb-4">
        <span className="w-1.5 h-1.5 rounded-full bg-[#4FD1C5]" />
        <p className="text-xs text-[#4FD1C5] font-mono tracking-widest font-semibold">
          {result === null ? 'PREDICT' : 'WHAT-IF'}
        </p>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
        {schema.fields.map((field) => (
          <div key={field.name}>
            <label className="text-xs text-[#8A93A6]">{field.name}</label>
            {field.type === 'categorical' && field.categories ? (
              <select
                value={values[field.name] ?? ''}
                onChange={(e) => handleChange(field.name, e.target.value)}
                className="w-full bg-[#121A2B] border-2 border-[#4FD1C5]/40 rounded-md px-3 py-2 text-sm text-[#EDEEF2] mt-1 focus:border-[#4FD1C5] transition"
              >
                <option value="" disabled>Select...</option>
                {field.categories.map((cat) => (
                  <option key={cat} value={cat}>{cat}</option>
                ))}
              </select>
            ) : (
              <input
                type="number"
                value={values[field.name] ?? ''}
                onChange={(e) => handleChange(field.name, e.target.value)}
                className="w-full bg-[#121A2B] border-2 border-[#2A3550] rounded-md px-3 py-2 text-sm text-[#EDEEF2] mt-1 focus:border-[#4FD1C5] transition"
              />
            )}
          </div>
        ))}
      </div>
      <button
        type="submit"
        disabled={predictMutation.isPending}
        className="mt-4 text-sm font-semibold bg-[#E8B04B] text-[#0B1220] hover:brightness-110 px-5 py-2.5 rounded-md disabled:opacity-50 transition"
      >
        {predictMutation.isPending ? 'Predicting...' : result === null ? 'Predict' : 'Re-predict'}
      </button>

      {predictMutation.isError && (
        <p className="text-[#E85D5D] text-xs mt-2">
          {(predictMutation.error as any)?.response?.data?.detail || 'Prediction failed.'}
        </p>
      )}

      {result !== null && (
        <div className="mt-4 bg-[#1A2436] border border-[#2A3550] rounded-lg p-4 text-center">
          {changed ? (
            <p className="font-mono text-lg font-bold">
              <span className="text-[#8A93A6] line-through text-sm">{previousResult}</span>
              {' → '}
              <span className={result > previousResult! ? 'text-[#4FD1C5]' : 'text-[#E85D5D]'}>{result}</span>
            </p>
          ) : (
            <p className="font-mono text-[#4FD1C5] text-lg font-bold">{result}</p>
          )}
          <p className="text-[10px] text-[#8A93A6] mt-1 tracking-wide">PREDICTED VALUE</p>
        </div>
      )}

      {explanation && (
        <div className="mt-3 bg-[#1A2436] border border-[#2A3550] rounded-lg p-4">
          <p className="text-[#E8B04B] text-xs font-semibold mb-2">Why this prediction</p>
          <ul className="space-y-1 text-[#EDEEF2] text-xs">
            {explanation.plain_english.map((sentence, i) => (
              <li key={i} className="flex gap-1.5">
                <span className="text-[#4FD1C5]">•</span>
                <span>{sentence}</span>
              </li>
            ))}
          </ul>
          <details className="mt-2">
            <summary className="text-[#8A93A6] text-xs cursor-pointer hover:text-[#EDEEF2]">
              Show verification
            </summary>
            <p className="text-[#8A93A6] text-xs mt-1 font-mono">
              base ({explanation.base_value}) + contributions ≈ {explanation.sum_check} (prediction: {result})
            </p>
          </details>
        </div>
      )}
    </form>
  )
}

function DatasetModels({ projectId, dataset }: { projectId: string; dataset: Dataset }) {
  const queryClient = useQueryClient()

  const { data: models } = useQuery({
    queryKey: ['models', dataset.id],
    queryFn: () => fetchModels(projectId, dataset.id),
    enabled: !!dataset.target_column,
  })

  const trainMutation = useMutation({
    mutationFn: () => trainModel(projectId, dataset.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models', dataset.id] })
    },
  })

  if (!dataset.target_column) return null

  return (
    <div className="mt-3">
      <button
        onClick={() => trainMutation.mutate()}
        disabled={trainMutation.isPending}
        className="text-sm font-semibold bg-[#4FD1C5] text-[#0B1220] hover:brightness-110 px-4 py-2 rounded-md transition disabled:opacity-50"
      >
        {trainMutation.isPending ? 'Training...' : 'Train Model'}
      </button>

      {trainMutation.isError && (
        <p className="text-[#E85D5D] text-xs mt-1">
          {(trainMutation.error as any)?.response?.data?.detail || 'Training failed.'}
        </p>
      )}

      {models && models.length > 0 && (
        <div className="mt-3 space-y-2">
          {models.map((m) => (
            <div key={m.id} className="bg-[#1A2436] border border-[#2A3550] rounded-lg p-4">
              <p className="text-[#E8B04B] font-semibold text-sm">🏆 Best Model: {m.model_name}</p>
              <MetricsDisplay metrics={m.metrics} />
              <FeatureImportanceDisplay projectId={projectId} datasetId={dataset.id} modelId={m.id} />
              <PredictionForm projectId={projectId} datasetId={dataset.id} modelId={m.id} />
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function ProjectDatasets({ projectId }: { projectId: string }) {
  const queryClient = useQueryClient()

  const { data: datasets, isLoading } = useQuery({
    queryKey: ['datasets', projectId],
    queryFn: () => fetchDatasets(projectId),
  })

  const uploadMutation = useMutation({
    mutationFn: (file: File) => uploadDataset(projectId, file),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['datasets', projectId] })
    },
  })

  const deleteMutation = useMutation({
    mutationFn: (datasetId: string) => deleteDataset(projectId, datasetId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['datasets', projectId] })
    },
  })

  const profileMutation = useMutation({
    mutationFn: (datasetId: string) => profileDataset(projectId, datasetId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['datasets', projectId] })
    },
  })

  const targetMutation = useMutation({
    mutationFn: ({ datasetId, column }: { datasetId: string; column: string }) =>
      setTargetColumn(projectId, datasetId, column),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['datasets', projectId] })
    },
  })

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (file) uploadMutation.mutate(file)
    e.target.value = ''
  }

  return (
    <div className="mt-4 pt-4 border-t border-[#2A3550]">
      <label className="inline-block">
        <span className="text-sm font-semibold bg-[#121A2B] border-2 border-[#4FD1C5]/50 hover:border-[#4FD1C5] hover:bg-[#1A2436] text-[#EDEEF2] px-6 py-3 rounded-lg cursor-pointer transition inline-block">
          {uploadMutation.isPending ? 'Uploading...' : '📤 Upload dataset'}
        </span>
        <input
          type="file"
          accept=".csv,.xls,.xlsx"
          onChange={handleFileChange}
          className="hidden"
          disabled={uploadMutation.isPending}
        />
      </label>

      {uploadMutation.isError && (
        <p className="text-[#E85D5D] text-sm mt-2">Upload failed — check file type.</p>
      )}

      {isLoading && <p className="text-[#8A93A6] text-sm mt-2">Loading datasets...</p>}

      {datasets && datasets.length > 0 && (
        <ul className="mt-3 space-y-3">
          {datasets.map((d) => (
            <li key={d.id} className="bg-[#0B1220] border border-[#2A3550] rounded-lg p-4 hover:border-[#4FD1C5]/40 transition-colors">
              <div className="flex items-center justify-between">
                <span className="text-sm text-[#EDEEF2]">📄 {d.filename}</span>
                <div className="flex gap-2">
                  <button
                    onClick={() => profileMutation.mutate(d.id)}
                    disabled={profileMutation.isPending}
                    className="text-xs bg-[#121A2B] border border-[#2A3550] hover:border-[#4FD1C5] text-[#EDEEF2] px-3 py-1.5 rounded transition"
                  >
                    {profileMutation.isPending ? 'Profiling...' : 'Profile'}
                  </button>
                  <button
                    onClick={() => {
                      if (confirm(`Delete "${d.filename}"? This cannot be undone.`)) {
                        deleteMutation.mutate(d.id)
                      }
                    }}
                    disabled={deleteMutation.isPending}
                    className="text-xs bg-[#2A1418] border border-[#E85D5D]/40 text-[#E85D5D] px-3 py-1.5 rounded hover:bg-[#3A1B20] transition"
                  >
                    Delete
                  </button>
                </div>
              </div>

              {d.profile_result && <QualityScoreBadge score={d.profile_result.quality_score} />}

              {d.profile_result && (
                <div className="mt-3 flex items-center gap-2 flex-wrap">
                  <label className="text-xs text-[#8A93A6]">Predict:</label>
                  <select
                    value={d.target_column ?? ''}
                    onChange={(e) => targetMutation.mutate({ datasetId: d.id, column: e.target.value })}
                    className="bg-[#121A2B] border-2 border-[#4FD1C5]/40 rounded-md px-3 py-2 text-sm text-[#EDEEF2] focus:border-[#4FD1C5] transition"
                  >
                    <option value="" disabled>Select column</option>
                    {Object.keys(d.profile_result.dtypes).map((col) => (
                      <option key={col} value={col}>{col}</option>
                    ))}
                  </select>
                  {d.problem_type && (
                    <span className="text-xs bg-[#1A2436] border border-[#4FD1C5]/30 text-[#4FD1C5] px-2.5 py-1.5 rounded">
                      {d.problem_type}
                    </span>
                  )}
                </div>
              )}

              <DatasetModels projectId={projectId} dataset={d} />
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

export default function DashboardPage() {
  const { getToken } = useAuth()
  const queryClient = useQueryClient()
  const [newProjectName, setNewProjectName] = useState('')

  useEffect(() => {
    setTokenGetter(getToken)
  }, [getToken])

  const { data: projects, isLoading, error } = useQuery({
    queryKey: ['projects'],
    queryFn: fetchProjects,
  })

  const createMutation = useMutation({
    mutationFn: createProject,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['projects'] })
      setNewProjectName('')
    },
  })

  const deleteMutation = useMutation({
    mutationFn: deleteProject,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['projects'] })
    },
  })

  function handleCreate(e: React.FormEvent) {
    e.preventDefault()
    if (newProjectName.trim()) createMutation.mutate(newProjectName.trim())
  }

  return (
    <div className="min-h-screen w-full bg-[#0B1220] text-[#EDEEF2] font-[Inter]">
      <header className="flex justify-between items-center px-6 md:px-10 py-5 border-b border-[#1A2436]">
        <div className="flex items-center gap-2">
          <span className="font-[Space_Grotesk] font-bold text-xl tracking-tight">ASK</span>
          <span className="hidden sm:inline text-[#8A93A6] text-sm">Analyze, Solve and Know</span>
        </div>
        <UserButton />
      </header>

      <main className="p-6 md:p-10 max-w-5xl mx-auto w-full">
        <form onSubmit={handleCreate} className="flex gap-2 mb-8">
          <input
            type="text"
            value={newProjectName}
            onChange={(e) => setNewProjectName(e.target.value)}
            placeholder="New project name"
            className="flex-1 bg-[#121A2B] border-2 border-[#2A3550] rounded-md px-4 py-2.5 text-[#EDEEF2] placeholder:text-[#8A93A6] focus:border-[#4FD1C5] transition"
          />
          <button
            type="submit"
            disabled={createMutation.isPending}
            className="bg-[#E8B04B] text-[#0B1220] font-semibold px-5 py-2.5 rounded-md hover:brightness-110 transition disabled:opacity-50"
          >
            {createMutation.isPending ? 'Creating...' : 'Create'}
          </button>
        </form>

        {isLoading && <p className="text-[#8A93A6]">Loading projects...</p>}
        {error && <p className="text-[#E85D5D]">Failed to load projects.</p>}

        <ul className="space-y-4">
          {projects?.map((project) => (
            <li key={project.id} className="bg-[#121A2B] border border-[#1A2436] rounded-xl px-6 py-5 shadow-sm hover:border-[#2A3550] transition-colors">
              <div className="flex justify-between items-center">
                <span className="font-[Space_Grotesk] font-semibold text-lg">{project.name}</span>
                <button
                  onClick={() => deleteMutation.mutate(project.id)}
                  disabled={deleteMutation.isPending}
                  className="text-[#E85D5D] hover:text-[#f08080] text-sm transition"
                >
                  Delete
                </button>
              </div>
              <ProjectDatasets projectId={project.id} />
            </li>
          ))}
        </ul>

        {projects?.length === 0 && (
          <p className="text-[#8A93A6] text-center mt-12">No projects yet — create your first one above.</p>
        )}
      </main>
    </div>
  )
}