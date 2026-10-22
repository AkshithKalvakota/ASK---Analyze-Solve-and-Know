import { useEffect, useState } from 'react'
import { isAxiosError } from 'axios'
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
  fetchSampleRow,
  type Dataset,
  type InputField,
  type PredictionExplanation,
  type Project,
  type SampleRow,
  type TrainedModel,
} from '../lib/api'

const MODEL_NAMES: Record<string, string> = {
  LinearRegression: 'Linear Regression',
  LogisticRegression: 'Logistic Regression',
  RandomForest: 'Random Forest',
  XGBoost: 'XGBoost',
}

function formatNumber(value: number) {
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 })
}

// Same rule as the backend's explanation sentences: whole numbers for big amounts
// (prices), one decimal for everyday sizes (MPa, ratings), two digits below 1
function formatAmount(value: number) {
  const size = Math.abs(value)
  if (size === 0) return '0'
  if (size >= 100) return Math.round(value).toLocaleString()
  if (size >= 1) return value.toLocaleString(undefined, { maximumFractionDigits: 1 })
  return value.toPrecision(2)
}

function formatPrediction(value: string | number) {
  return typeof value === 'number' ? formatNumber(value) : value
}

// Prefer the server's own message (it explains what to do); fall back to a friendly default
function errorMessage(error: unknown, fallback: string) {
  const detail = isAxiosError(error) ? error.response?.data?.detail : undefined
  return typeof detail === 'string' ? detail : fallback
}

// "smoker_yes" -> "smoker (yes)", using the categories from the input schema
function friendlyFeature(feature: string, fields: InputField[] = []) {
  for (const field of fields) {
    for (const category of field.categories ?? []) {
      if (feature === `${field.name}_${category}`) return `${field.name} (${category})`
    }
  }
  return feature
}

type TargetCheck = { level: 'block' | 'warn'; message: string }

// Spot target columns that can't sensibly be predicted, using the stored profile.
// Datasets profiled before unique_counts existed simply get no check.
function checkTarget(column: string, profile: NonNullable<Dataset['profile_result']>): TargetCheck | null {
  const unique = profile.unique_counts?.[column]
  if (unique === undefined) return null
  const rows = profile.n_rows
  const dtype = profile.dtypes[column] ?? ''
  const isText = !/int|float|bool/.test(dtype)
  const missing: number = profile.missing_by_column?.[column]?.missing_count ?? 0

  if (unique <= 1) {
    return { level: 'block', message: 'Every row has the same value here, so there is nothing to predict. Choose another column.' }
  }
  if (isText && unique > 20 && unique > rows / 2) {
    return {
      level: 'block',
      message: `Almost every row has a different value (${unique.toLocaleString()} of ${rows.toLocaleString()}), like a name or an ID. A model can't learn categories from that. Choose another column.`,
    }
  }
  if (/int/.test(dtype) && unique === rows) {
    return { level: 'warn', message: 'Every row has a different whole number here. If this is an ID or a row number, the model will learn nothing useful.' }
  }
  if (isText && unique > 20) {
    return { level: 'warn', message: `This column has ${unique} different values. Categories that appear only once or twice can't be learned, so building may fail.` }
  }
  if (missing > 0) {
    return { level: 'warn', message: `${missing.toLocaleString()} ${missing === 1 ? 'row has' : 'rows have'} no value in this column. ASK will leave ${missing === 1 ? 'it' : 'them'} out when building the model.` }
  }
  return null
}

// Ticks once a second while `active`, for elapsed-time and slow-loading messages
function useNow(active: boolean) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (!active) return
    const id = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(id)
  }, [active])
  return now
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-2 mb-3">
      <span className="w-1.5 h-1.5 rounded-full bg-[#4FD1C5]" />
      <p className="text-xs text-[#4FD1C5] font-mono tracking-widest font-semibold uppercase">{children}</p>
    </div>
  )
}

function DataSummary({ profile }: { profile: NonNullable<Dataset['profile_result']> }) {
  const score = profile.quality_score
  const color =
    score >= 80 ? 'bg-[#4FD1C5] text-[#0B1220]' : score >= 50 ? 'bg-[#E8B04B] text-[#0B1220]' : 'bg-[#E85D5D] text-white'
  const label = score >= 80 ? 'Good' : score >= 50 ? 'Fair' : 'Poor'

  const outliers = Object.values((profile.outliers_by_column ?? {}) as Record<string, number>).reduce((a, b) => a + b, 0)
  const duplicates: number = profile.duplicate_count ?? 0
  const missingPct: number = profile.missing_pct_total ?? 0
  const facts = [
    `${profile.n_rows.toLocaleString()} rows`,
    `${profile.n_columns} columns`,
    missingPct > 0 ? `${missingPct}% of cells empty` : 'no empty cells',
    duplicates > 0 ? `${duplicates} duplicate ${duplicates === 1 ? 'row' : 'rows'}` : 'no duplicate rows',
    `${outliers.toLocaleString()} unusual ${outliers === 1 ? 'value' : 'values'}`,
  ]

  return (
    <div className="flex items-center gap-3 mt-3">
      <div className={`${color} font-mono text-xs font-bold rounded-full w-10 h-10 shrink-0 flex items-center justify-center`}>
        {Math.round(score)}
      </div>
      <div>
        <p className="text-xs text-[#8A93A6]">
          Data quality: <span className="text-[#EDEEF2]">{label}</span>
        </p>
        <p className="text-xs text-[#8A93A6] mt-0.5">{facts.join(' · ')}</p>
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
    <div className="grid grid-cols-3 gap-2 mt-3">
      {Object.entries(metrics).filter(([key]) => key in labels).map(([key, value]) => (
        <div key={key} className="bg-[#0B1220] border border-[#2A3550] rounded p-2 text-center">
          <p className="text-[10px] text-[#8A93A6] uppercase tracking-wide">{labels[key] ?? key}</p>
          <p className="font-mono text-sm font-semibold text-[#4FD1C5] mt-1">
            {isPercentMetric(key) ? `${(value * 100).toFixed(1)}%` : key === 'r2' ? value.toFixed(4) : formatNumber(value)}
          </p>
        </div>
      ))}
    </div>
  )
}

function ModelSummary({ model, problemType, target }: { model: TrainedModel; problemType: string; target: string }) {
  const isRegression = problemType === 'regression'
  const score = isRegression ? model.metrics.r2 : model.metrics.f1
  const rating = isRegression
    ? score >= 0.8 ? 'Very good' : score >= 0.6 ? 'Good' : score >= 0.4 ? 'Fair' : 'Weak'
    : score >= 0.85 ? 'Very good' : score >= 0.7 ? 'Good' : score >= 0.55 ? 'Fair' : 'Weak'
  const scoreKey = isRegression ? 'r2' : 'f1'
  const tried = Object.entries(model.all_results).sort(([, a], [, b]) => b[scoreKey] - a[scoreKey])

  return (
    <div>
      <p className="text-[#E8B04B] font-semibold text-sm">🏆 Best method: {MODEL_NAMES[model.model_name] ?? model.model_name}</p>
      <div className="mt-3 bg-[#0B1220] border border-[#2A3550] rounded-lg p-3 text-sm space-y-1">
        {/* The rating and the winner both come from R² (numbers) or F1 (categories),
            never plain accuracy, which looks good on lopsided data even for a useless model */}
        <p>
          Model quality: <span className="font-semibold text-[#4FD1C5]">{rating}</span>{' '}
          <span className="text-xs text-[#8A93A6]">(based on {isRegression ? 'R² score' : 'F1 score'})</span>
        </p>
        {isRegression ? (
          <>
            <p className="text-[#8A93A6]">
              R² {model.metrics.r2.toFixed(2)}: explains about {Math.max(0, Math.round(model.metrics.r2 * 100))}% of the ups and downs in {target}.
            </p>
            <p className="text-[#8A93A6]">
              Its guesses are usually off by about <span className="text-[#EDEEF2]">{formatAmount(model.metrics.mae)}</span>.
            </p>
          </>
        ) : (
          <p className="text-[#8A93A6]">
            F1 {model.metrics.f1.toFixed(2)}: balances catching the real cases with avoiding false alarms, so it isn't
            fooled when one {target} value is rare.
          </p>
        )}
      </div>
      <details className="mt-2">
        <summary className="text-[#8A93A6] text-xs cursor-pointer hover:text-[#EDEEF2]">Technical details</summary>
        <MetricsDisplay metrics={model.metrics} />
        <p className="text-xs text-[#8A93A6] mt-3 mb-1">
          We tried {tried.length} methods ({isRegression ? 'R² score' : 'F1 score'}, higher is better):
        </p>
        <ul className="text-xs space-y-0.5">
          {tried.map(([name, metrics]) => (
            <li key={name} className={name === model.model_name ? 'text-[#4FD1C5]' : 'text-[#8A93A6]'}>
              {name === model.model_name ? '★' : '·'} {MODEL_NAMES[name] ?? name}: {(metrics[scoreKey] * 100).toFixed(1)}%
            </li>
          ))}
        </ul>
      </details>
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
  const { data: schema } = useQuery({
    queryKey: ['input-schema', modelId],
    queryFn: () => fetchInputSchema(projectId, datasetId, modelId),
  })

  if (isLoading) return <p className="text-xs text-[#8A93A6] mt-4">Working out what matters most...</p>
  if (!data || data.feature_importance.length === 0) return null

  const maxImportance = Math.max(...data.feature_importance.map((f) => f.importance))

  return (
    <div className="mt-5">
      <SectionLabel>What matters most</SectionLabel>
      <p className="text-xs text-[#8A93A6] -mt-1 mb-3">The inputs that change the prediction the most, biggest first.</p>
      <div className="space-y-2">
        {data.feature_importance.slice(0, 8).map((f) => {
          const name = friendlyFeature(f.feature, schema?.fields)
          return (
            <div key={f.feature} className="flex items-center gap-3">
              <span className="text-xs text-[#EDEEF2] w-36 truncate" title={name}>{name}</span>
              <div className="flex-1 bg-[#0B1220] rounded h-3">
                <div
                  className="bg-[#4FD1C5] h-3 rounded"
                  style={{ width: `${maxImportance > 0 ? (f.importance / maxImportance) * 100 : 0}%` }}
                />
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// Shown after predicting a real row: the true value from the file next to the prediction
function RealRowCheck({
  actual,
  predicted,
  rowNumber,
  isCategory,
}: {
  actual: SampleRow['actual']
  predicted: string | number
  rowNumber: number
  isCategory: boolean // category answers (even numbers like a 1-5 rating) are right or wrong, not "off by"
}) {
  let verdict: React.ReactNode
  if (actual === null) {
    verdict = <p className="text-[#8A93A6]">Row {rowNumber} of your file has no real value to compare with.</p>
  } else if (!isCategory && typeof actual === 'number' && typeof predicted === 'number') {
    const off = Math.abs(predicted - actual)
    const pct = actual !== 0 ? Math.round((off / Math.abs(actual)) * 100) : null
    verdict = (
      <>
        <p>
          Real value in row {rowNumber} of your file: <span className="font-mono font-semibold text-[#EDEEF2]">{formatNumber(actual)}</span>
        </p>
        <p className="text-[#8A93A6] mt-0.5">The model was off by {formatAmount(off)}{pct !== null ? ` (${pct}%)` : ''}.</p>
      </>
    )
  } else {
    const right = String(actual) === String(predicted)
    verdict = (
      <p>
        Real value in row {rowNumber} of your file: <span className="font-semibold text-[#EDEEF2]">{String(actual)}</span>{' '}
        <span className={right ? 'text-[#4FD1C5]' : 'text-[#E85D5D]'}>
          {right ? '✓ The model got this one right.' : `✗ The model said ${String(predicted)}.`}
        </span>
      </p>
    )
  }
  return (
    <div className="mt-3 pt-3 border-t border-[#2A3550] text-xs text-left">
      {verdict}
      <p className="text-[10px] text-[#8A93A6] mt-1">This row was hidden from the model while it learned, so it's a fair test.</p>
    </div>
  )
}

function PredictionForm({
  projectId,
  datasetId,
  modelId,
  target,
  metrics,
}: {
  projectId: string
  datasetId: string
  modelId: string
  target: string
  metrics: Record<string, number> // the model's own scores: R² means it predicts numbers
}) {
  const [values, setValues] = useState<Record<string, string>>({})
  const [result, setResult] = useState<string | number | null>(null)
  const [previousResult, setPreviousResult] = useState<string | number | null>(null)
  const [explanation, setExplanation] = useState<PredictionExplanation | null>(null)

  const { data: schema } = useQuery({
    queryKey: ['input-schema', modelId],
    queryFn: () => fetchInputSchema(projectId, datasetId, modelId),
  })

  // Start every field at a typical value from the training data, so the user only
  // changes what they care about and nothing is ever sent empty
  const [prefilledFor, setPrefilledFor] = useState<string | null>(null)
  if (schema && prefilledFor !== modelId) {
    setPrefilledFor(modelId)
    const defaults: Record<string, string> = {}
    for (const field of schema.fields) {
      if (field.default !== null && field.default !== undefined) defaults[field.name] = String(field.default)
    }
    setValues(defaults)
  }

  // Prediction and explanation use the same snapshot of the form (passed as the
  // mutation variables), so editing a box mid-request can't mix two inputs; they
  // also run in parallel instead of one after the other
  const predictMutation = useMutation({
    mutationFn: async (inputs: Record<string, string>) => {
      const [prediction, explanation] = await Promise.all([
        predict(projectId, datasetId, modelId, inputs),
        explainPrediction(projectId, datasetId, modelId, inputs).catch(() => null),
      ])
      return { prediction: prediction.prediction, explanation }
    },
    onSuccess: (data) => {
      setPreviousResult(result)
      setResult(data.prediction)
      setExplanation(data.explanation)
    },
  })

  // "Try a real row": fill the form from a row the model never saw, predict it,
  // and show the true value next to the prediction
  const [realRow, setRealRow] = useState<{ actual: SampleRow['actual']; rowNumber: number } | null>(null)
  const sampleRowMutation = useMutation({
    mutationFn: () => fetchSampleRow(projectId, datasetId, modelId),
    onSuccess: (row) => {
      const filled: Record<string, string> = {}
      for (const [name, value] of Object.entries(row.values)) filled[name] = value === null ? '' : String(value)
      setValues(filled)
      setRealRow({ actual: row.actual, rowNumber: row.row_number })
      setResult(null)
      setPreviousResult(null)
      setExplanation(null)
      predictMutation.mutate(filled)
    },
  })

  function handleChange(fieldName: string, value: string) {
    setValues((prev) => ({ ...prev, [fieldName]: value }))
    setRealRow(null) // edited inputs no longer match the real row's true value
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    predictMutation.mutate(values)
  }

  if (!schema) return <p className="text-xs text-[#8A93A6] mt-3">Loading input fields...</p>

  const prefilled = schema.fields.some((field) => field.default !== null && field.default !== undefined)
  const changed = previousResult !== null && result !== null && previousResult !== result
  const numericChange =
    changed && typeof result === 'number' && typeof previousResult === 'number' ? result - previousResult : null
  // Typical miss on the hidden test rows (models trained before this was saved have none)
  const typicalMiss = metrics.error_p50
  const largeMiss = metrics.error_p90

  return (
    <form onSubmit={handleSubmit} className="mt-5 bg-[#0B1220] border border-[#2A3550] rounded-lg p-4">
      <SectionLabel>{result === null ? 'Try a prediction' : 'What if…?'}</SectionLabel>
      <p className="text-xs text-[#8A93A6] -mt-1 mb-4">
        {result !== null
          ? 'Change any value and press Update to see how the prediction moves.'
          : prefilled
            ? `Each box starts at a typical value from your data. Change what you like, then press Predict ${target}.`
            : `Fill in every box, then press Predict ${target}. (Rebuild the model to get boxes pre-filled with typical values.)`}
      </p>
      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
        {schema.fields.map((field) => {
          const inputId = `${modelId}-${field.name}`
          return (
            <div key={field.name}>
              <label htmlFor={inputId} className="text-xs text-[#8A93A6]">{field.name}</label>
              {field.type === 'categorical' && field.categories ? (
                <select
                  id={inputId}
                  required
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
                <>
                  <input
                    id={inputId}
                    type="number"
                    required
                    step={field.step === 1 ? 1 : 'any'}
                    value={values[field.name] ?? ''}
                    onChange={(e) => handleChange(field.name, e.target.value)}
                    className="w-full bg-[#121A2B] border-2 border-[#2A3550] rounded-md px-3 py-2 text-sm text-[#EDEEF2] mt-1 focus:border-[#4FD1C5] transition"
                  />
                  {field.min != null && field.max != null && (
                    <p className="text-[10px] text-[#8A93A6] mt-1">
                      Usually {formatNumber(field.min)}–{formatNumber(field.max)}
                    </p>
                  )}
                </>
              )}
            </div>
          )
        })}
      </div>
      <div className="mt-4 flex flex-wrap items-center gap-3">
        <button
          type="submit"
          disabled={predictMutation.isPending}
          className="text-sm font-semibold bg-[#E8B04B] text-[#0B1220] hover:brightness-110 px-5 py-2.5 rounded-md disabled:opacity-50 transition"
        >
          {predictMutation.isPending ? 'Predicting...' : result === null ? `Predict ${target}` : 'Update prediction'}
        </button>
        <button
          type="button"
          onClick={() => sampleRowMutation.mutate()}
          disabled={sampleRowMutation.isPending || predictMutation.isPending}
          className="text-sm bg-[#121A2B] border-2 border-[#4FD1C5]/50 hover:border-[#4FD1C5] text-[#EDEEF2] px-4 py-2 rounded-md transition disabled:opacity-50"
        >
          {sampleRowMutation.isPending ? 'Picking a row...' : '🎲 Try a real row from my data'}
        </button>
      </div>

      {predictMutation.isError && (
        <p className="text-[#E85D5D] text-xs mt-2">
          {errorMessage(predictMutation.error, 'Prediction failed. Please check the values and try again.')}
        </p>
      )}
      {sampleRowMutation.isError && (
        <p className="text-[#E85D5D] text-xs mt-2">
          {errorMessage(sampleRowMutation.error, "Couldn't pick a row from your file. Please try again.")}
        </p>
      )}

      {result !== null && (
        <div className="mt-4 bg-[#1A2436] border border-[#2A3550] rounded-lg p-4 text-center">
          {changed ? (
            <p className="font-mono text-lg font-bold">
              <span className="text-[#8A93A6] line-through text-sm">{formatPrediction(previousResult!)}</span>
              {' → '}
              <span className={numericChange === null || numericChange > 0 ? 'text-[#4FD1C5]' : 'text-[#E85D5D]'}>
                {formatPrediction(result)}
              </span>
            </p>
          ) : (
            <p className="font-mono text-[#4FD1C5] text-lg font-bold">{formatPrediction(result)}</p>
          )}
          {numericChange !== null && (
            <p className={`text-xs mt-1 ${numericChange > 0 ? 'text-[#4FD1C5]' : 'text-[#E85D5D]'}`}>
              {numericChange > 0 ? '▲' : '▼'} {formatNumber(Math.abs(numericChange))} {numericChange > 0 ? 'higher' : 'lower'} than before
            </p>
          )}
          <p className="text-[10px] text-[#8A93A6] mt-1 tracking-wide uppercase">Predicted {target}</p>
          {typeof result === 'number' && typicalMiss != null && largeMiss != null && (
            <p className="text-xs text-[#8A93A6] mt-2">
              Likely between {formatAmount(result - typicalMiss)} and {formatAmount(result + typicalMiss)}.
              <span className="block text-[10px] mt-0.5">
                Half of the model's test predictions were this close; 9 in 10 were within ±{formatAmount(largeMiss)}.
              </span>
            </p>
          )}
          {realRow && (
            <RealRowCheck
              actual={realRow.actual}
              predicted={result}
              rowNumber={realRow.rowNumber}
              isCategory={!('r2' in metrics)}
            />
          )}
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
              Show the maths
            </summary>
            <p className="text-[#8A93A6] text-xs mt-1 font-mono">
              average ({formatNumber(explanation.base_value)}) + each input's effect ≈ {formatNumber(explanation.sum_check)} (prediction: {formatPrediction(result!)})
            </p>
          </details>
        </div>
      )}
    </form>
  )
}

function DatasetModels({
  projectId,
  dataset,
  targetCheck,
}: {
  projectId: string
  dataset: Dataset
  targetCheck: TargetCheck | null
}) {
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

  const now = useNow(trainMutation.isPending)
  const elapsed = Math.max(0, Math.round((now - trainMutation.submittedAt) / 1000))

  if (!dataset.target_column) return null
  // Spreadsheet headers often carry stray spaces ("strength(MPa) "); trim for display only
  const target = dataset.target_column.trim()
  const blocked = targetCheck?.level === 'block'

  return (
    <div className="mt-3">
      <button
        onClick={() => trainMutation.mutate()}
        disabled={trainMutation.isPending || blocked}
        className="text-sm font-semibold bg-[#4FD1C5] text-[#0B1220] hover:brightness-110 px-4 py-2 rounded-md transition disabled:opacity-50"
      >
        {trainMutation.isPending ? `Building your model... ${elapsed}s` : models?.length ? 'Rebuild my model' : 'Build my model'}
      </button>
      {trainMutation.isPending && (
        <p className="text-xs text-[#8A93A6] mt-1">Cleaning the data, trying 3 methods and keeping the best. Usually 5–30 seconds.</p>
      )}
      {!trainMutation.isPending && !models?.length && !blocked && (
        <p className="text-xs text-[#8A93A6] mt-1">Next: build a model that predicts {target}.</p>
      )}

      {trainMutation.isError && (
        <p className="text-[#E85D5D] text-xs mt-1">
          {errorMessage(trainMutation.error, 'Building the model failed. Please try again.')}
        </p>
      )}

      {models && models.length > 0 && (
        <div className="mt-3 space-y-2">
          {models.map((m) => (
            <div key={m.id} className="bg-[#1A2436] border border-[#2A3550] rounded-lg p-4">
              <ModelSummary model={m} problemType={dataset.problem_type ?? ''} target={target} />
              <FeatureImportanceDisplay projectId={projectId} datasetId={dataset.id} modelId={m.id} />
              <PredictionForm projectId={projectId} datasetId={dataset.id} modelId={m.id} target={target} metrics={m.metrics} />
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

  const profileMutation = useMutation({
    mutationFn: (datasetId: string) => profileDataset(projectId, datasetId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['datasets', projectId] })
    },
  })

  // Check the data straight after upload: there is nothing else useful to do first
  const uploadMutation = useMutation({
    mutationFn: (file: File) => uploadDataset(projectId, file),
    onSuccess: (dataset) => {
      queryClient.invalidateQueries({ queryKey: ['datasets', projectId] })
      profileMutation.mutate(dataset.id)
    },
  })

  const deleteMutation = useMutation({
    mutationFn: (datasetId: string) => deleteDataset(projectId, datasetId),
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
      <label className="inline-flex items-center gap-3 flex-wrap">
        <span className="text-sm font-semibold bg-[#121A2B] border-2 border-[#4FD1C5]/50 hover:border-[#4FD1C5] hover:bg-[#1A2436] text-[#EDEEF2] px-6 py-3 rounded-lg cursor-pointer transition inline-block">
          {uploadMutation.isPending ? 'Uploading...' : '📤 Upload a spreadsheet'}
        </span>
        <span className="text-xs text-[#8A93A6]">CSV or Excel (.csv, .xlsx, .xls)</span>
        <input
          type="file"
          accept=".csv,.xls,.xlsx"
          onChange={handleFileChange}
          className="hidden"
          disabled={uploadMutation.isPending}
        />
      </label>

      {uploadMutation.isError && (
        <p className="text-[#E85D5D] text-sm mt-2">
          {errorMessage(uploadMutation.error, 'Upload failed. Please use a spreadsheet saved as .csv, .xlsx or .xls.')}
        </p>
      )}

      {isLoading && <p className="text-[#8A93A6] text-sm mt-2">Loading datasets...</p>}

      {datasets && datasets.length > 0 && (
        <ul className="mt-3 space-y-3">
          {datasets.map((d) => {
            const checking = profileMutation.isPending && profileMutation.variables === d.id
            const checkFailed = profileMutation.isError && profileMutation.variables === d.id
            const settingTarget = targetMutation.isPending && targetMutation.variables?.datasetId === d.id
            const targetFailed = targetMutation.isError && targetMutation.variables?.datasetId === d.id
            const columns = d.profile_result?.columns ?? Object.keys(d.profile_result?.dtypes ?? {})
            const selectId = `${d.id}-target`
            const targetCheck = d.profile_result && d.target_column ? checkTarget(d.target_column, d.profile_result) : null

            return (
              <li key={d.id} className="bg-[#0B1220] border border-[#2A3550] rounded-lg p-4 hover:border-[#4FD1C5]/40 transition-colors">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-sm text-[#EDEEF2] truncate">📄 {d.filename}</span>
                  <div className="flex gap-2 shrink-0">
                    <button
                      onClick={() => profileMutation.mutate(d.id)}
                      disabled={checking}
                      className="text-xs bg-[#121A2B] border border-[#2A3550] hover:border-[#4FD1C5] text-[#EDEEF2] px-3 py-1.5 rounded transition"
                    >
                      {checking ? 'Checking...' : d.profile_result ? 'Re-check data' : 'Check my data'}
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

                {checking && <p className="text-xs text-[#8A93A6] mt-3">Checking your data for gaps, duplicates and unusual values...</p>}
                {checkFailed && (
                  <p className="text-[#E85D5D] text-xs mt-3">
                    {errorMessage(profileMutation.error, "We couldn't check this file. Please try again.")}
                  </p>
                )}
                {!checking && !checkFailed && !d.profile_result && (
                  <p className="text-xs text-[#8A93A6] mt-3">Next: check your data so ASK can see what's inside.</p>
                )}

                {d.profile_result && <DataSummary profile={d.profile_result} />}

                {d.profile_result && (
                  <div className="mt-4">
                    <label htmlFor={selectId} className="block text-xs text-[#8A93A6] mb-1">
                      What do you want to predict?
                    </label>
                    <div className="flex items-center gap-2 flex-wrap">
                      <select
                        id={selectId}
                        value={d.target_column ?? ''}
                        disabled={settingTarget}
                        onChange={(e) => targetMutation.mutate({ datasetId: d.id, column: e.target.value })}
                        className="bg-[#121A2B] border-2 border-[#4FD1C5]/40 rounded-md px-3 py-2 text-sm text-[#EDEEF2] focus:border-[#4FD1C5] transition"
                      >
                        <option value="" disabled>Choose a column</option>
                        {columns.map((col) => (
                          <option key={col} value={col}>
                            {col}
                            {checkTarget(col, d.profile_result!)?.level === 'block' ? " (can't be predicted)" : ''}
                          </option>
                        ))}
                      </select>
                      {settingTarget && <span className="text-xs text-[#8A93A6]">Looking at this column...</span>}
                      {!settingTarget && d.problem_type && (
                        <span className="text-xs bg-[#1A2436] border border-[#4FD1C5]/30 text-[#4FD1C5] px-2.5 py-1.5 rounded">
                          {d.problem_type === 'regression' ? 'Predicting a number' : 'Predicting a category'}
                        </span>
                      )}
                    </div>
                    {targetFailed && (
                      <p className="text-[#E85D5D] text-xs mt-1">
                        {errorMessage(targetMutation.error, "Couldn't use that column. Please try another one.")}
                      </p>
                    )}
                    {!settingTarget && targetCheck && (
                      <p
                        className={`text-xs mt-2 rounded-md px-3 py-2 border ${
                          targetCheck.level === 'block'
                            ? 'text-[#E85D5D] bg-[#2A1418] border-[#E85D5D]/40'
                            : 'text-[#E8B04B] bg-[#2A2414] border-[#E8B04B]/40'
                        }`}
                      >
                        {targetCheck.level === 'block' ? '⛔ ' : '⚠️ '}
                        {targetCheck.message}
                      </p>
                    )}
                  </div>
                )}

                <DatasetModels projectId={projectId} dataset={d} targetCheck={targetCheck} />
              </li>
            )
          })}
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

  // The free backend sleeps when idle; after a few seconds of loading, say so
  const [openedAt] = useState(() => Date.now())
  const now = useNow(isLoading)
  const slowStart = isLoading && now - openedAt > 4000

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

  // "Try it with sample data": run the whole flow on the bundled insurance file
  // using the normal API calls, telling the user which step is running
  const [sampleStep, setSampleStep] = useState<string | null>(null)
  const sampleMutation = useMutation({
    mutationFn: async () => {
      setSampleStep('Creating a sample project...')
      const project = await createProject('Sample: insurance charges')
      queryClient.invalidateQueries({ queryKey: ['projects'] })

      setSampleStep('Uploading insurance.csv (1,338 people)...')
      const response = await fetch('/samples/insurance.csv')
      if (!response.ok) throw new Error('Sample file missing')
      const file = new File([await response.blob()], 'insurance.csv', { type: 'text/csv' })
      const dataset = await uploadDataset(project.id, file)

      setSampleStep('Checking the data...')
      await profileDataset(project.id, dataset.id)
      setSampleStep('Choosing what to predict: charges...')
      await setTargetColumn(project.id, dataset.id, 'charges')
      setSampleStep('Building the model (trying 3 methods, about 10 seconds)...')
      await trainModel(project.id, dataset.id)
      return { project, dataset }
    },
    onSuccess: ({ project, dataset }) => {
      queryClient.invalidateQueries({ queryKey: ['datasets', project.id] })
      queryClient.invalidateQueries({ queryKey: ['models', dataset.id] })
      setTimeout(() => document.getElementById(`project-${project.id}`)?.scrollIntoView({ behavior: 'smooth' }), 400)
    },
    onSettled: () => setSampleStep(null),
  })

  function handleCreate(e: React.FormEvent) {
    e.preventDefault()
    if (newProjectName.trim()) createMutation.mutate(newProjectName.trim())
  }

  function handleDeleteProject(project: Project) {
    if (confirm(`Delete the project "${project.name}" and all its files and models? This cannot be undone.`)) {
      deleteMutation.mutate(project.id)
    }
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
        <form onSubmit={handleCreate} className="flex gap-2">
          <label htmlFor="new-project-name" className="sr-only">New project name</label>
          <input
            id="new-project-name"
            type="text"
            value={newProjectName}
            onChange={(e) => setNewProjectName(e.target.value)}
            placeholder="Name a new project, e.g. House prices"
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
        <div className="mt-3 mb-8 flex flex-wrap items-center gap-x-3 gap-y-1">
          <button
            type="button"
            onClick={() => sampleMutation.mutate()}
            disabled={sampleMutation.isPending}
            className="text-sm font-semibold text-[#4FD1C5] hover:underline disabled:opacity-60 disabled:no-underline"
          >
            ✨ Try it with sample data
          </button>
          <span className="text-xs text-[#8A93A6]">
            {sampleStep ?? 'Loads 1,338 insurance records and builds a model that predicts medical charges.'}
          </span>
          {sampleMutation.isError && (
            <p className="basis-full text-xs text-[#E85D5D]">
              {errorMessage(sampleMutation.error, "The sample didn't finish. Please try again.")}
            </p>
          )}
        </div>

        {isLoading && (
          <div className="text-[#8A93A6]">
            <p>Loading projects...</p>
            {slowStart && (
              <p className="text-sm mt-1">
                Waking up the server. After a quiet spell the first visit can take up to a minute. Thanks for waiting!
              </p>
            )}
          </div>
        )}
        {error && <p className="text-[#E85D5D]">Couldn't reach the server. Please refresh the page in a minute.</p>}

        <ul className="space-y-4">
          {projects?.map((project) => (
            <li key={project.id} id={`project-${project.id}`} className="bg-[#121A2B] border border-[#1A2436] rounded-xl px-6 py-5 shadow-sm hover:border-[#2A3550] transition-colors">
              <div className="flex justify-between items-center">
                <span className="font-[Space_Grotesk] font-semibold text-lg">{project.name}</span>
                <button
                  onClick={() => handleDeleteProject(project)}
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
          <div className="text-center mt-12 text-[#8A93A6]">
            <p>No projects yet.</p>
            <p className="text-sm mt-1">Give your first project a name above, then upload a spreadsheet to get started.</p>
            <p className="text-sm mt-4">New here? See the whole thing work in about 20 seconds:</p>
            <button
              type="button"
              onClick={() => sampleMutation.mutate()}
              disabled={sampleMutation.isPending}
              className="mt-2 bg-[#E8B04B] text-[#0B1220] font-semibold px-5 py-2.5 rounded-md hover:brightness-110 transition disabled:opacity-50"
            >
              {sampleMutation.isPending ? 'Setting up the sample...' : '✨ Try it with sample data'}
            </button>
          </div>
        )}
      </main>
    </div>
  )
}
