'use client'

import { useState } from 'react'
import { ChevronDown } from 'lucide-react'
import { RequestMetadata } from '@/lib/types'
import { formatDuration } from '@/lib/utils'

interface RequestDetailsProps {
  metadata: RequestMetadata | null
}

export function RequestDetails({ metadata }: RequestDetailsProps) {
  const [expanded, setExpanded] = useState(false)

  if (!metadata) return null

  const fields = [
    { label: 'Input Tokens', value: metadata.input_tokens?.toString() },
    { label: 'Output Tokens', value: metadata.output_tokens?.toString() },
    {
      label: 'Prefill Latency',
      value: metadata.prefill_latency
        ? formatDuration(metadata.prefill_latency)
        : undefined,
    },
    {
      label: 'TTFT',
      value: metadata.ttft ? formatDuration(metadata.ttft) : undefined,
    },
    {
      label: 'Mean ITL',
      value: metadata.mean_itl ? formatDuration(metadata.mean_itl) : undefined,
    },
    {
      label: 'Total Latency',
      value: metadata.total_latency
        ? formatDuration(metadata.total_latency)
        : undefined,
    },
    { label: 'Reached EOS', value: metadata.reached_eos?.toString() },
    { label: 'Model', value: metadata.model },
    { label: 'Quantization', value: metadata.quantization },
  ]

  const hasData = fields.some((f) => f.value)
  if (!hasData) return null

  return (
    <div className="rounded-card border border-slate-200 bg-white">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-6 py-4 flex items-center justify-between text-left transition-colors hover:bg-slate-50"
      >
        <div>
          <h3 className="font-bold text-navy">Request Details</h3>
        </div>
        <ChevronDown
          className={`h-5 w-5 text-slate-blue transition-transform ${expanded ? 'rotate-180' : ''}`}
        />
      </button>

      {expanded && (
        <div className="border-t border-slate-100 px-6 py-4">
          <div className="space-y-2 text-sm">
            {fields.map((field) => (
              <div key={field.label} className="flex justify-between py-2">
                <span className="text-slate-blue">{field.label}:</span>
                <span className="font-medium text-navy">{field.value || '—'}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
