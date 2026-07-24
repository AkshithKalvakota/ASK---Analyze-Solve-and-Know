import { useEffect } from 'react'
import { useAuth, UserButton } from '@clerk/clerk-react'
import { setTokenGetter } from '../lib/api'

export default function DashboardPage() {
  const { getToken } = useAuth()

  useEffect(() => {
    setTokenGetter(getToken)
  }, [getToken])

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      <header className="flex justify-between items-center p-4 border-b border-gray-800">
        <h1 className="text-xl font-semibold">ASK - Analyze, Solve and Know</h1>
        <UserButton />
      </header>
      <main className="p-8">
        <p className="text-2xl">Welcome to your dashboard 🎉</p>
      </main>
    </div>
  )
}