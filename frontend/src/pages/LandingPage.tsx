import { SignInButton } from '@clerk/clerk-react'

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-gray-950 text-white flex flex-col items-center justify-center gap-4">
      <h1 className="text-3xl font-semibold">ASK - Analyze, Solve and Know </h1>
      <p className="text-gray-400">Sign in to access your dashboard.</p>
      <SignInButton mode="modal" />
    </div>
  )
}