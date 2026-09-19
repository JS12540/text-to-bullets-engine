'use client'

import { FileText, Github, Heart } from 'lucide-react'

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
          <div className="flex flex-col items-start gap-2 sm:items-end">
            <p className="text-xs text-slate-blue">
              Built for learning and real-world inference.
            </p>
            <p className="flex items-center gap-1 text-xs text-slate-blue">
              Made with <Heart className="h-3.5 w-3.5 fill-red-500 text-red-500" />
            </p>
            <div className="flex items-center gap-4">
              <a
                href="https://github.com/JS12540/text-to-bullets-engine"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1.5 text-xs text-slate-blue transition-colors hover:text-navy"
              >
                <span>GitHub repo</span>
                <Github className="h-3.5 w-3.5" />
              </a>
              <a
                href="https://huggingface.co/JayShah07/falconai-text-bullet-t5"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1.5 text-xs text-slate-blue transition-colors hover:text-navy"
              >
                <span>Hugging Face model</span>
                <span className="text-sm leading-none">🤗</span>
              </a>
            </div>
          </div>
        </div>
      </div>
    </footer>
  )
}
