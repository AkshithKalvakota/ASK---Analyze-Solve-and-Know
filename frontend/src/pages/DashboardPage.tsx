import { UserButton } from '@clerk/clerk-react'

export default function DashboardPage() {
  return (
    <div className="min-h-screen bg-gray-950 text-white">
      <header className="flex justify-between items-center p-4 border-b border-gray-800">
        <h1 className="text-xl font-semibold">AI Data Analyst</h1>
        <UserButton />
      </header>
      <main className="p-8">
        <p className="text-2xl">Welcome to your dashboard 🎉</p>
      </main>
    </div>
  )
}