export function cn(...classes: (string | undefined | false)[]): string {
  return classes.filter(Boolean).join(' ')
}

export function formatDuration(ms: number): string {
  if (ms < 1000) {
    return `${Math.round(ms)} ms`
  }
  return `${(ms / 1000).toFixed(2)} s`
}

export function formatMetricValue(value: number | undefined, unit?: string): string {
  if (value === undefined || value === null) {
    return '—'
  }
  if (unit === 'ms') {
    return `${Math.round(value)} ms`
  }
  if (unit === 's') {
    return `${(value / 1000).toFixed(2)} s`
  }
  return value.toString()
}

export function getMetricLabel(metric: string): string {
  const labels: Record<string, string> = {
    total_latency: 'Total Latency',
    ttft: 'Time to First Token',
    mean_itl: 'Mean Inter-Token Latency',
    input_tokens: 'Input Tokens',
    output_tokens: 'Output Tokens',
    prefill_time: 'Prefill Latency',
  }
  return labels[metric] || metric
}
