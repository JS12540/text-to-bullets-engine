'use client'

import { FileText, Zap, Shield, Cpu, Github } from 'lucide-react'

export function Header() {
  return (
    <header className="w-full border-b border-slate-200 bg-white">
      <div className="mx-auto max-w-container px-6 py-6 sm:py-8">
        <div className="flex items-start justify-between gap-6">
          <div className="flex items-start gap-4">
            <div className="text-primary-blue pt-1">
              <FileText className="h-6 w-6" />
            </div>
            <div className="flex-1">
              <h1 className="text-2xl font-bold text-navy">Text to Bullets</h1>
              <p className="mt-1 text-sm text-slate-blue">
                Turn long, unstructured text into clear, concise bullet points
                using a fine-tuned T5 model.
              </p>
            </div>
          </div>

          <div className="flex flex-col items-end gap-3 sm:flex-row sm:items-center sm:gap-4">
            <div className="flex items-center gap-4">
              <a
                href="https://github.com/JS12540/text-to-bullets-engine"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1.5 text-sm text-slate-blue transition-colors hover:text-navy"
              >
                <span>GitHub repo</span>
                <Github className="h-4 w-4" />
              </a>
              <a
                href="https://huggingface.co/JayShah07/falconai-text-bullet-t5"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1.5 text-sm text-slate-blue transition-colors hover:text-navy"
              >
                <span>Hugging Face model</span>
                <span className="text-base leading-none">🤗</span>
              </a>
            </div>
            <div className="flex flex-wrap justify-end gap-2">
              <div className="flex items-center gap-1 rounded-full bg-slate-50 px-3 py-1.5 text-xs text-slate-blue">
                <Zap className="h-3.5 w-3.5" />
                <span>Fast</span>
              </div>
              <div className="flex items-center gap-1 rounded-full bg-slate-50 px-3 py-1.5 text-xs text-slate-blue">
                <Shield className="h-3.5 w-3.5" />
                <span>Private</span>
              </div>
              <div className="flex items-center gap-1 rounded-full bg-slate-50 px-3 py-1.5 text-xs text-slate-blue">
                <Cpu className="h-3.5 w-3.5" />
                <span>CPU Optimized</span>
              </div>
            </div>
            <div className="rounded-full bg-indigo/10 px-3 py-1.5 text-xs font-medium text-indigo">
              AI Powered
            </div>
          </div>
        </div>
      </div>
    </header>
  )
}
