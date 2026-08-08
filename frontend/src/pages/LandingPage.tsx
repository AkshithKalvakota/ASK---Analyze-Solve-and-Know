import { useEffect, useState } from 'react'
import { SignInButton } from '@clerk/clerk-react'

const QUESTION = 'What will this house sell for?'
const ANSWER = '₹52,00,000 — driven mainly by area and number of bathrooms.'

function TypedQuestion() {
  const [typed, setTyped] = useState('')
  const [showAnswer, setShowAnswer] = useState(false)

  useEffect(() => {
    const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (prefersReduced) {
      setTyped(QUESTION)
      setShowAnswer(true)
      return
    }

    let i = 0
    const typeInterval = setInterval(() => {
      i++
      setTyped(QUESTION.slice(0, i))
      if (i >= QUESTION.length) {
        clearInterval(typeInterval)
        setTimeout(() => setShowAnswer(true), 400)
      }
    }, 45)

    return () => clearInterval(typeInterval)
  }, [])

  return (
    <div className="w-full max-w-xl">
      <div className="flex items-center gap-2 font-mono text-sm text-[#4FD1C5]">
        <span className="text-[#E8B04B]">ASK&gt;</span>
        <span>{typed}</span>
        <span className="inline-block w-[2px] h-4 bg-[#4FD1C5] animate-pulse" />
      </div>

      <div
        className={`mt-6 transition-all duration-700 ${
          showAnswer ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2 pointer-events-none'
        }`}
      >
        <div className="bg-[#1A2436] border border-[#2A3550] rounded-lg p-4 flex items-start gap-4">
          <svg width="56" height="32" viewBox="0 0 56 32" className="shrink-0 mt-1">
            <polyline
              points="0,10 10,8 20,14 30,6 40,22 50,18"
              fill="none"
              stroke="#E8B04B"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          <p className="text-[#EDEEF2] text-sm leading-relaxed font-[Inter]">{ANSWER}</p>
        </div>
      </div>
    </div>
  )
}

function PipelineStep({ label, detail }: { label: string; detail: string }) {
  return (
    <div className="flex-1 min-w-[140px]">
      <p className="font-mono text-xs text-[#4FD1C5] tracking-wider">{label}</p>
      <p className="text-[#8A93A6] text-sm mt-1">{detail}</p>
    </div>
  )
}

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-[#0B1220] text-[#EDEEF2] flex flex-col font-[Inter]">
      <header className="flex justify-between items-center px-6 md:px-10 py-6 border-b border-[#1A2436]">
        <div className="flex items-center gap-2">
          <span className="font-[Space_Grotesk] font-bold text-xl tracking-tight">ASK</span>
          <span className="hidden sm:inline text-[#8A93A6] text-sm">Analyze, Solve and Know</span>
        </div>
        <SignInButton mode="modal">
          <button className="bg-[#E8B04B] text-[#0B1220] font-semibold text-sm px-4 py-2 rounded-md hover:brightness-110 transition">
            Sign in
          </button>
        </SignInButton>
      </header>

      <main className="flex-1 flex flex-col items-center justify-center px-6 py-16 gap-16">
        <div className="text-center max-w-2xl">
          <h1 className="font-[Space_Grotesk] font-bold text-4xl md:text-5xl tracking-tight leading-tight">
            Upload your data.
            <br />
            <span className="text-[#4FD1C5]">Ask it anything.</span>
          </h1>
          <p className="mt-4 text-[#8A93A6] text-lg max-w-lg mx-auto">
            ASK profiles your dataset, trains the right model automatically, and explains every
            prediction in plain language — no data science degree required.
          </p>
        </div>

        <TypedQuestion />

        <SignInButton mode="modal">
          <button className="bg-[#E8B04B] text-[#0B1220] font-semibold px-6 py-3 rounded-md hover:brightness-110 transition text-base">
            Get started 
          </button>
        </SignInButton>

        <div className="w-full max-w-3xl border-t border-[#1A2436] pt-8 flex flex-wrap gap-6">
          <PipelineStep label="UPLOAD" detail="CSV or Excel, any shape" />
          <PipelineStep label="PROFILE" detail="Quality score, gaps, outliers" />
          <PipelineStep label="TRAIN" detail="Best model picked automatically" />
          <PipelineStep label="ASK" detail="Predict, explain, what-if" />
        </div>
      </main>

      <footer className="text-center text-[#8A93A6] text-xs py-6 border-t border-[#1A2436]">
        ASK — Analyze, Solve and Know
      </footer>
    </div>
  )
}