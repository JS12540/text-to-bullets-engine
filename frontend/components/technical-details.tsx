'use client'

import {
  Clock,
  Zap,
  RefreshCw,
  ArrowDownToLine,
  ArrowUpFromLine,
  Cpu,
} from 'lucide-react'
import { MetricCard } from './metric-card'
import { RequestDetails } from './request-details'
import { GenerationMetrics, RequestMetadata } from '@/lib/types'
import { formatDuration } from '@/lib/utils'

interface TechnicalDetailsProps {
  metrics: GenerationMetrics | null
  metadata: RequestMetadata | null
}

export function TechnicalDetails({ metrics, metadata }: TechnicalDetailsProps) {
  const metricsList = [
    {
      icon: Clock,
      label: 'Total Latency',
      value: metrics?.total_latency ? formatDuration(metrics.total_latency) : '—',
    },
    {
      icon: Zap,
      label: 'Time to First Token',
      value: metrics?.ttft ? formatDuration(metrics.ttft) : '—',
    },
    {
      icon: RefreshCw,
      label: 'Mean Inter-Token Latency',
      value: metrics?.mean_itl ? formatDuration(metrics.mean_itl) : '—',
    },
    {
      icon: ArrowDownToLine,
      label: 'Input Tokens',
      value: metrics?.input_tokens?.toString() ?? '—',
    },
    {
      icon: ArrowUpFromLine,
      label: 'Output Tokens',
      value: metrics?.output_tokens?.toString() ?? '—',
    },
    {
      icon: Cpu,
      label: 'Model',
      value: 'T5 INT8',
      extra: 'CPU • TorchAO',
    },
  ]

  return (
    <div className="rounded-card border border-slate-200 bg-white shadow-subtle">
      <div className="border-b border-slate-100 px-6 py-4 sm:px-6 sm:py-5">
        <h2 className="text-lg font-bold text-navy">Technical Details</h2>
        <p className="mt-2 text-sm text-slate-blue">
          Inference metrics and model information for this request.
        </p>
      </div>

      <div className="px-6 py-5 sm:px-6">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {metricsList.map((metric) => (
            <div key={metric.label}>
              {metric.extra ? (
                <div className="rounded-card border border-slate-200 bg-white px-4 py-4 sm:px-6 sm:py-6">
                  <div className="flex items-start gap-3">
                    <div className="text-slate-blue pt-0.5">
                      <metric.icon className="h-5 w-5" />
                    </div>
                    <div className="flex-1">
                      <p className="text-xs font-medium text-slate-blue uppercase tracking-wide">
                        {metric.label}
                      </p>
                      <p className="mt-2 text-lg font-bold text-navy sm:text-xl">
                        {metric.value}
                      </p>
                      <p className="mt-1 text-xs text-slate-blue">{metric.extra}</p>
                    </div>
                  </div>
                </div>
              ) : (
                <MetricCard icon={metric.icon} label={metric.label} value={metric.value} />
              )}
            </div>
          ))}
        </div>
      </div>

      <div className="border-t border-slate-100 px-6 py-4 sm:px-6">
        <RequestDetails metadata={metadata} />
      </div>
    </div>
  )
}
