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
    score >= 80 ? 'bg-green-600' : score >= 50 ? 'bg-yellow-600' : 'bg-red-600'
  const label = score >= 80 ? 'Good' : score >= 50 ? 'Fair' : 'Poor'

  return (
    <div className="flex items-center gap-2 mt-2">
      <div className={`${color} text-white text-xs font-bold rounded-full w-10 h-10 flex items-center justify-center`}>
        {Math.round(score)}
      </div>
      <div>
        <p className="text-xs text-gray-400">
          Data Quality: <span className="text-gray-200">{label}</span>
        </p>
      </div>
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
    <div className="grid grid-cols-3 gap-2 mt-2">
      {Object.entries(metrics).map(([key, value]) => (
        <div key={key} className="bg-gray-900 border border-gray-800 rounded p-2 text-center">
          <p className="text-xs text-gray-500">{labels[key] ?? key}</p>
          <p className="text-sm font-semibold text-white mt-1">
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

  if (isLoading) return <p className="text-xs text-gray-500 mt-2">Loading feature importance...</p>
  if (!data || data.feature_importance.length === 0) return null

  const maxImportance = Math.max(...data.feature_importance.map((f) => f.importance))

  return (
    <div className="mt-3">
      <p className="text-xs text-gray-400 mb-1">Feature Importance:</p>
      <div className="space-y-1">
        {data.feature_importance.slice(0, 8).map((f) => (
          <div key={f.feature} className="flex items-center gap-2">
            <span className="text-xs text-gray-500 w-20 truncate">{f.feature}</span>
            <div className="flex-1 bg-gray-800 rounded h-3">
              <div
                className="bg-blue-600 h-3 rounded"
                style={{ width: `${(f.importance / maxImportance) * 100}%` }}
              />
            </div>
            <span className="text-xs text-gray-500 w-12 text-right">{f.importance}</span>
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

  if (!schema) return <p className="text-xs text-gray-500 mt-2">Loading input fields...</p>

  const changed =
    previousResult !== null && result !== null && previousResult !== result

  return (
    <form onSubmit={handleSubmit} className="mt-3 bg-gray-950 border border-gray-800 rounded p-3">
      <p className="text-xs text-gray-400 mb-2">
        {result === null ? 'Enter values to get a prediction:' : 'Adjust values to see what changes (what-if):'}
      </p>
      <div className="grid grid-cols-2 gap-2">
        {schema.fields.map((field) => (
          <div key={field.name}>
            <label className="text-xs text-gray-500">{field.name}</label>
            {field.type === 'categorical' && field.categories ? (
              <select
                value={values[field.name] ?? ''}
                onChange={(e) => handleChange(field.name, e.target.value)}
                className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs"
              >
                <option value="" disabled>
                  Select...
                </option>
                {field.categories.map((cat) => (
                  <option key={cat} value={cat}>
                    {cat}
                  </option>
                ))}
              </select>
            ) : (
              <input
                type="number"
                value={values[field.name] ?? ''}
                onChange={(e) => handleChange(field.name, e.target.value)}
                className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs"
              />
            )}
          </div>
        ))}
      </div>
      <button
        type="submit"
        disabled={predictMutation.isPending}
        className="mt-3 text-xs bg-blue-700 hover:bg-blue-600 px-3 py-1.5 rounded"
      >
        {predictMutation.isPending
          ? 'Predicting...'
          : result === null
          ? 'Predict'
          : 'Re-predict (what-if)'}
      </button>

      {predictMutation.isError && (
        <p className="text-red-400 text-xs mt-2">
          {(predictMutation.error as any)?.response?.data?.detail || 'Prediction failed.'}
        </p>
      )}

      {result !== null && (
        <div className="mt-3 bg-gray-900 border border-green-900/40 rounded-lg p-3 text-center">
          {changed ? (
            <p className="text-lg font-bold">
              <span className="text-gray-500 line-through text-sm">{previousResult}</span>
              {' → '}
              <span className={result > previousResult! ? 'text-green-400' : 'text-red-400'}>
                {result}
              </span>
            </p>
          ) : (
            <p className="text-green-400 text-lg font-bold">{result}</p>
          )}
          <p className="text-xs text-gray-500 mt-1">Predicted Value</p>
        </div>
      )}

      {explanation && (
        <div className="mt-3 bg-blue-950/30 border border-blue-900/40 rounded-lg p-3">
          <p className="text-blue-300 text-xs font-semibold mb-2">💡 Why this prediction</p>
          <ul className="space-y-1 text-gray-300 text-xs">
            {explanation.plain_english.map((sentence, i) => (
              <li key={i} className="flex gap-1">
                <span>•</span>
                <span>{sentence}</span>
              </li>
            ))}
          </ul>
          <details className="mt-2">
            <summary className="text-gray-600 text-xs cursor-pointer hover:text-gray-400">
              Show verification
            </summary>
            <p className="text-gray-600 text-xs mt-1">
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
    <div className="mt-2">
      <button
        onClick={() => trainMutation.mutate()}
        disabled={trainMutation.isPending}
        className="text-xs bg-green-800 hover:bg-green-700 px-2 py-1 rounded"
      >
        {trainMutation.isPending ? 'Training...' : 'Train Model'}
      </button>

      {trainMutation.isError && (
        <p className="text-red-400 text-xs mt-1">
          {(trainMutation.error as any)?.response?.data?.detail || 'Training failed.'}
        </p>
      )}

      {models && models.length > 0 && (
        <div className="mt-2 space-y-2">
          {models.map((m) => (
            <div key={m.id} className="bg-gray-900 border border-purple-900/40 rounded-lg p-3 text-xs">
              <div className="flex items-center justify-between mb-1">
                <p className="text-purple-300 font-semibold text-sm">🏆 Best Model: {m.model_name}</p>
              </div>
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
    if (file) {
      uploadMutation.mutate(file)
    }
    e.target.value = ''
  }

  return (
    <div className="mt-3 pt-3 border-t border-gray-800">
      <label className="inline-block">
        <span className="text-sm bg-gray-800 hover:bg-gray-700 px-3 py-1.5 rounded cursor-pointer">
          {uploadMutation.isPending ? 'Uploading...' : 'Upload dataset'}
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
        <p className="text-red-400 text-sm mt-2">Upload failed — check file type.</p>
      )}

      {isLoading && <p className="text-gray-500 text-sm mt-2">Loading datasets...</p>}

      {datasets && datasets.length > 0 && (
        <ul className="mt-3 space-y-3">
          {datasets.map((d) => (
            <li
              key={d.id}
              className="text-sm text-gray-400 bg-gray-950 border border-gray-800 rounded-lg p-4"
            >
              <div className="flex items-center justify-between">
                <span>📄 {d.filename}</span>
                <div className="flex gap-2">
                  <button
                    onClick={() => profileMutation.mutate(d.id)}
                    disabled={profileMutation.isPending}
                    className="text-xs bg-gray-800 hover:bg-gray-700 px-2 py-1 rounded"
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
                    className="text-xs bg-red-900 hover:bg-red-800 text-red-200 px-2 py-1 rounded"
                  >
                    Delete
                  </button>
                </div>
              </div>

              {d.profile_result && (
                <QualityScoreBadge score={d.profile_result.quality_score} />
              )}

              {d.profile_result && (
                <div className="mt-2 flex items-center gap-2">
                  <label className="text-xs">Predict:</label>
                  <select
                    value={d.target_column ?? ''}
                    onChange={(e) =>
                      targetMutation.mutate({ datasetId: d.id, column: e.target.value })
                    }
                    className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs"
                  >
                    <option value="" disabled>
                      Select column
                    </option>
                    {Object.keys(d.profile_result.dtypes).map((col) => (
                      <option key={col} value={col}>
                        {col}
                      </option>
                    ))}
                  </select>
                  {d.problem_type && (
                    <span className="text-xs bg-purple-900 text-purple-200 px-2 py-1 rounded">
                      Detected: {d.problem_type}
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
    if (newProjectName.trim()) {
      createMutation.mutate(newProjectName.trim())
    }
  }

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      <header className="flex justify-between items-center p-4 border-b border-gray-800">
        <h1 className="text-xl font-semibold">ASK - Analyze, Solve and Know</h1>
        <UserButton />
      </header>

      <main className="p-8 max-w-2xl mx-auto">
        <form onSubmit={handleCreate} className="flex gap-2 mb-6">
          <input
            type="text"
            value={newProjectName}
            onChange={(e) => setNewProjectName(e.target.value)}
            placeholder="New project name"
            className="flex-1 bg-gray-900 border border-gray-700 rounded px-3 py-2 text-white"
          />
          <button
            type="submit"
            disabled={createMutation.isPending}
            className="bg-purple-600 hover:bg-purple-700 px-4 py-2 rounded disabled:opacity-50"
          >
            {createMutation.isPending ? 'Creating...' : 'Create'}
          </button>
        </form>

        {isLoading && <p className="text-gray-400">Loading projects...</p>}
        {error && <p className="text-red-500">Failed to load projects.</p>}

        <ul className="space-y-2">
          {projects?.map((project) => (
            <li
              key={project.id}
              className="bg-gray-900 border border-gray-800 rounded-lg px-5 py-4 shadow-sm"
            >
              <div className="flex justify-between items-center">
                <span>{project.name}</span>
                <button
                  onClick={() => deleteMutation.mutate(project.id)}
                  disabled={deleteMutation.isPending}
                  className="text-red-400 hover:text-red-300 text-sm"
                >
                  Delete
                </button>
              </div>
              <ProjectDatasets projectId={project.id} />
            </li>
          ))}
        </ul>

        {projects?.length === 0 && (
          <p className="text-gray-500 text-center mt-8">No projects yet — create your first one above.</p>
        )}
      </main>
    </div>
  )
}