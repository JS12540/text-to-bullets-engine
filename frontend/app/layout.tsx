import type { Metadata, Viewport } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Text to Bullets',
  description: 'Turn long, unstructured text into clear, concise bullet points using a fine-tuned T5 model.',
}

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="antialiased bg-cool-white">
        {children}
      </body>
    </html>
  )
}
