import { useEffect, useState } from 'react'
import { useAuth, UserButton } from '@clerk/clerk-react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { setTokenGetter, fetchProjects, createProject, deleteProject } from '../lib/api'

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
              className="flex justify-between items-center bg-gray-900 border border-gray-800 rounded px-4 py-3"
            >
              <span>{project.name}</span>
              <button
                onClick={() => deleteMutation.mutate(project.id)}
                disabled={deleteMutation.isPending}
                className="text-red-400 hover:text-red-300 text-sm"
              >
                Delete
              </button>
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