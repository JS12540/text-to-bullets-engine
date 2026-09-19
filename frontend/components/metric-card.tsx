'use client'

import { LucideIcon } from 'lucide-react'

interface MetricCardProps {
  icon: LucideIcon
  label: string
  value: string | number
  unit?: string
}

export function MetricCard({ icon: Icon, label, value, unit }: MetricCardProps) {
  return (
    <div className="rounded-card border border-slate-200 bg-white px-4 py-4 sm:px-6 sm:py-6">
      <div className="flex items-start gap-3">
        <div className="text-slate-blue pt-0.5">
          <Icon className="h-5 w-5" />
        </div>
        <div className="flex-1">
          <p className="text-xs font-medium text-slate-blue uppercase tracking-wide">
            {label}
          </p>
          <p className="mt-2 text-xl font-bold text-navy sm:text-2xl">
            {value}
            {unit && <span className="text-sm font-normal text-slate-blue ml-1">{unit}</span>}
          </p>
        </div>
      </div>
    </div>
  )
}
