'use client'

import { useState } from 'react'
import { FileText, Sparkles, ArrowRight, Settings2, ChevronDown } from 'lucide-react'

interface TextInputCardProps {
  value: string
  onChange: (value: string) => void
  onGenerate: () => void
  onExample: () => void
  isLoading: boolean
  isEmpty: boolean
}

// Backend enforces MAX_INPUT_TOKENS=2048; system+task prompt adds ~84 tokens on top of
// whatever's typed here (not counted in this char count), leaving headroom below 2048.
// This is a token-budget ceiling, not a quality guarantee — the model is a small
// single-document summarizer and degrades on inputs mixing unrelated topics well
// before hitting this limit (see the note below the textarea).
const CHARACTER_LIMIT = 8000

export function TextInputCard({
  value,
  onChange,
  onGenerate,
  onExample,
  isLoading,
  isEmpty,
}: TextInputCardProps) {
  const [showAdvanced, setShowAdvanced] = useState(false)
  const characterCount = value.length

  return (
    <div className="rounded-card border border-slate-200 bg-white shadow-subtle">
      <div className="border-b border-slate-100 px-6 py-4 sm:px-6 sm:py-5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <FileText className="h-5 w-5 text-primary-blue" />
            <h2 className="text-lg font-bold text-navy">Input</h2>
          </div>
          <button
            onClick={onExample}
            className="flex items-center gap-2 rounded-md bg-slate-50 px-3 py-1.5 text-sm font-medium text-slate-blue transition-colors hover:bg-slate-100"
          >
            <FileText className="h-4 w-4" />
            <span>Example</span>
          </button>
        </div>
        <p className="mt-2 text-sm text-slate-blue">
          Paste your text below. The model will convert it into clear, concise bullet points.
        </p>
        <p className="mt-1 text-xs text-slate-blue">
          Best results with a single topic or document — mixing multiple unrelated topics in one request can produce garbled output.
        </p>
      </div>

      <div className="px-6 py-5 sm:px-6">
        <textarea
          value={value}
          onChange={(e) => onChange(e.currentTarget.value.slice(0, CHARACTER_LIMIT))}
          placeholder="Paste the text you want to convert into bullet points..."
          className="w-full min-h-[300px] rounded-input border border-slate-200 bg-white px-4 py-3 text-base leading-relaxed text-navy placeholder-slate-400 focus:border-primary-blue focus:outline-none focus:ring-2 focus:ring-primary-blue/20 resize-y"
          aria-label="Input text"
        />
        <p className="mt-2 text-right text-xs text-slate-blue">
          {characterCount.toLocaleString()} / {CHARACTER_LIMIT.toLocaleString()}
        </p>
      </div>

      <div className="border-t border-slate-100">
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="w-full px-6 py-3 flex items-center gap-2 text-sm font-medium text-slate-blue transition-colors hover:bg-slate-50"
        >
          <Settings2 className="h-4 w-4" />
          <span>Advanced options</span>
          <ChevronDown
            className={`h-4 w-4 ml-auto transition-transform ${showAdvanced ? 'rotate-180' : ''}`}
          />
        </button>

        {showAdvanced && (
          <div className="border-t border-slate-100 px-6 py-4 bg-slate-50">
            <div className="space-y-3 text-sm">
              <div className="flex justify-between">
                <span className="text-slate-blue">Maximum output tokens:</span>
                <span className="font-medium text-navy">Backend controlled</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-blue">Decoding:</span>
                <span className="font-medium text-navy">Greedy</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-blue">Streaming:</span>
                <span className="font-medium text-navy">Enabled</span>
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="px-6 py-4 sm:px-6">
        <button
          onClick={onGenerate}
          disabled={isEmpty || isLoading}
          className="w-full flex items-center justify-center gap-2 rounded-md bg-primary-blue px-6 py-3 font-medium text-white transition-colors hover:bg-blue-700 disabled:bg-slate-200 disabled:text-slate-400 disabled:cursor-not-allowed h-12 sm:h-13"
          aria-label="Generate bullets"
        >
          <Sparkles className="h-5 w-5" />
          <span>
            {isLoading
              ? value.length > 0
                ? 'Streaming...'
                : 'Generating...'
              : 'Generate Bullets'}
          </span>
          <ArrowRight className="h-5 w-5" />
        </button>
      </div>
    </div>
  )
}
