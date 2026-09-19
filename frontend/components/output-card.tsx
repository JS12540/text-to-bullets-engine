'use client'

import { useState, useEffect } from 'react'
import { List, Check, Copy, CheckCircle2 } from 'lucide-react'
import { GenerationStatus } from '@/lib/types'

interface OutputCardProps {
  output: string
  status: GenerationStatus
  totalLatency?: number
}

export function OutputCard({ output, status, totalLatency }: OutputCardProps) {
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    if (copied) {
      const timer = setTimeout(() => setCopied(false), 2000)
      return () => clearTimeout(timer)
    }
    return
  }, [copied])

  const handleCopy = () => {
    navigator.clipboard.writeText(output)
    setCopied(true)
  }

  const isComplete = status === 'completed'
  const isStreaming = status === 'streaming'

  return (
    <div className="rounded-card border border-slate-200 bg-white shadow-subtle">
      <div className="border-b border-slate-100 px-6 py-4 sm:px-6 sm:py-5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <List className="h-5 w-5 text-primary-blue" />
            <h2 className="text-lg font-bold text-navy">Output</h2>
          </div>
          <div className="flex items-center gap-2">
            {isComplete && totalLatency !== undefined && (
              <div className="flex items-center gap-1 rounded-full bg-green-50 px-3 py-1.5 text-xs font-medium text-success-green">
                <CheckCircle2 className="h-3.5 w-3.5" />
                <span>Completed in {(totalLatency / 1000).toFixed(2)}s</span>
              </div>
            )}
            <button
              onClick={handleCopy}
              disabled={!output}
              className="rounded-md bg-slate-50 px-3 py-1.5 text-sm font-medium text-slate-blue transition-colors hover:bg-slate-100 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              aria-label="Copy output"
            >
              {copied ? (
                <>
                  <Check className="h-4 w-4" />
                  <span>Copied</span>
                </>
              ) : (
                <>
                  <Copy className="h-4 w-4" />
                  <span>Copy</span>
                </>
              )}
            </button>
          </div>
        </div>
        <p className="mt-2 text-sm text-slate-blue">
          Generated bullet points from your input text.
        </p>
      </div>

      <div className="min-h-[300px] px-6 py-5 sm:px-6">
        {output ? (
          <div className="prose prose-sm max-w-none text-navy" aria-live="polite">
            {output.split('\n').map((line, i) => {
              if (!line.trim()) return <div key={i} className="h-2" />
              if (line.trim().startsWith('-')) {
                return (
                  <div key={i} className="flex gap-3 py-1">
                    <span className="font-bold text-primary-blue flex-shrink-0">–</span>
                    <span className="text-navy">{line.substring(1).trim()}</span>
                  </div>
                )
              }
              return (
                <p key={i} className="py-1 text-navy">
                  {line}
                </p>
              )
            })}
            {isStreaming && (
              <span
                className="ml-1 inline-block h-5 w-1 bg-primary-blue animate-pulse"
                aria-label="Streaming"
              />
            )}
          </div>
        ) : (
          <p className="text-slate-blue italic">
            Your generated bullet points will appear here.
          </p>
        )}
      </div>
    </div>
  )
}
