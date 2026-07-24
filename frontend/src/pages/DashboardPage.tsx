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
  profileDataset,
  setTargetColumn,
} from '../lib/api'

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
        <ul className="mt-2 space-y-3">
          {datasets.map((d) => (
            <li key={d.id} className="text-sm text-gray-400">
              <div className="flex items-center justify-between">
                <span>📄 {d.filename}</span>
                <button
                  onClick={() => profileMutation.mutate(d.id)}
                  disabled={profileMutation.isPending}
                  className="text-xs bg-gray-800 hover:bg-gray-700 px-2 py-1 rounded"
                >
                  {profileMutation.isPending ? 'Profiling...' : 'Profile'}
                </button>
              </div>

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
              className="bg-gray-900 border border-gray-800 rounded px-4 py-3"
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