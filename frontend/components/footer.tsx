'use client'

import { FileText } from 'lucide-react'

export function Footer() {
  return (
    <footer className="w-full border-t border-slate-200 bg-white">
      <div className="mx-auto max-w-container px-6 py-6 sm:py-8">
        <div className="flex flex-col items-start justify-between gap-4 sm:flex-row sm:items-center">
          <div className="flex items-center gap-3">
            <FileText className="h-5 w-5 text-primary-blue" />
            <div>
              <p className="text-sm font-medium text-navy">Text to Bullets</p>
              <p className="text-xs text-slate-blue">Clearer text. Better understanding.</p>
            </div>
          </div>
          <p className="text-xs text-slate-blue">
            Built for learning and real-world inference.
          </p>
        </div>
      </div>
    </footer>
  )
}
