'use client'

import { useState, useEffect, useRef } from 'react'
import { AlertCircle } from 'lucide-react'
import { Header } from '@/components/header'
import { TextInputCard } from '@/components/text-input-card'
import { OutputCard } from '@/components/output-card'
import { TechnicalDetails } from '@/components/technical-details'
import { Footer } from '@/components/footer'
import { streamBullets } from '@/lib/stream'
import { getApiBaseUrl, checkBackendReady } from '@/lib/api'
import { GenerationStatus, GenerationMetrics, RequestMetadata } from '@/lib/types'

export default function Home() {
  const [inputText, setInputText] = useState('')
  const [outputText, setOutputText] = useState('')
  const [status, setStatus] = useState<GenerationStatus>('idle')
  const [error, setError] = useState<string | null>(null)
  const [metrics, setMetrics] = useState<GenerationMetrics | null>(null)
  const [metadata, setMetadata] = useState<RequestMetadata | null>(null)
  const [backendReady, setBackendReady] = useState(true)
  const [temperature, setTemperature] = useState(0)
  const abortControllerRef = useRef<AbortController | null>(null)
  const startTimeRef = useRef<number>(0)

  const apiBaseUrl = getApiBaseUrl()

  // Check backend health on mount
  useEffect(() => {
    checkBackendReady(apiBaseUrl)
      .then(setBackendReady)
      .catch(() => setBackendReady(false))
  }, [apiBaseUrl])

  const handleGenerate = async () => {
    if (!inputText.trim()) return

    setError(null)
    setOutputText('')
    setMetrics(null)
    setMetadata(null)
    setStatus('connecting')
    startTimeRef.current = Date.now()

    abortControllerRef.current = new AbortController()

    try {
      const result = await streamBullets(
        inputText,
        apiBaseUrl,
        (chunk) => {
          setOutputText((prev) => prev + chunk)
          setStatus('streaming')
        },
        (newMetrics) => {
          setMetrics(newMetrics)
          // Convert metrics to metadata for display
          const totalLatency = Date.now() - startTimeRef.current
          setMetadata({
            input_tokens: newMetrics.input_tokens,
            output_tokens: newMetrics.output_tokens,
            ttft: newMetrics.ttft,
            mean_itl: newMetrics.mean_itl,
            total_latency: newMetrics.total_latency || totalLatency,
            prefill_latency: newMetrics.prefill_time,
            reached_eos: newMetrics.reached_eos,
            model: 'T5 INT8',
            quantization: 'TorchAO',
          })
        },
        abortControllerRef.current.signal,
        temperature
      )

      setOutputText(result.text)
      setMetrics(result.metrics)

      // Calculate total latency
      const totalLatency = Date.now() - startTimeRef.current
      setMetadata({
        input_tokens: result.metrics.input_tokens,
        output_tokens: result.metrics.output_tokens,
        ttft: result.metrics.ttft,
        mean_itl: result.metrics.mean_itl,
        total_latency: result.metrics.total_latency || totalLatency,
        prefill_latency: result.metrics.prefill_time,
        reached_eos: result.metrics.reached_eos,
        model: 'T5 INT8',
        quantization: 'TorchAO',
      })

      setStatus('completed')
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred'

      if (errorMessage.includes('token')) {
        setError(
          `Input exceeds maximum length. Please reduce your text and try again.`
        )
      } else if (errorMessage === 'Request cancelled') {
        setStatus('idle')
        return
      } else {
        setError(
          `Unable to generate bullet points. ${
            errorMessage.includes('Backend')
              ? 'Please check that the inference server is running.'
              : `Error: ${errorMessage}`
          }`
        )
      }

      setStatus('error')
    }
  }

  const handleExample = () => {
    setInputText(
      `Acme reported quarterly revenue of $4.2 billion, up 12% year over year. Operating profit increased 8% to $620 million, although operating margin declined from 17.2% to 14.8%. The company added 1.3 million customers during the quarter and raised full-year revenue guidance from $16 billion to $17.5 billion. Management warned that European demand weakened in July.`
    )
  }

  const isEmpty = !inputText.trim()
  const isLoading = status === 'connecting' || status === 'streaming'

  return (
    <div className="min-h-screen flex flex-col bg-cool-white">
      <Header />

      <main className="flex-1 mx-auto w-full max-w-container px-6 py-8 sm:py-12">
        {error && (
          <div className="mb-6 rounded-card border border-red-200 bg-red-50 p-4 flex gap-3">
            <AlertCircle className="h-5 w-5 text-red-600 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-medium text-red-900">Unable to generate bullet points</p>
              <p className="mt-1 text-xs text-red-800">{error}</p>
            </div>
          </div>
        )}

        {!backendReady && (
          <div className="mb-6 rounded-card border border-amber-200 bg-amber-50 p-4 flex gap-3">
            <AlertCircle className="h-5 w-5 text-amber-600 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-medium text-amber-900">Backend server unavailable</p>
              <p className="mt-1 text-xs text-amber-800">
                Please ensure the inference server is running at {apiBaseUrl}
              </p>
            </div>
          </div>
        )}

        <div className="grid gap-6 lg:grid-cols-2">
          <TextInputCard
            value={inputText}
            onChange={setInputText}
            onGenerate={handleGenerate}
            onExample={handleExample}
            isLoading={isLoading}
            isEmpty={isEmpty}
            temperature={temperature}
            onTemperatureChange={setTemperature}
          />
          <OutputCard output={outputText} status={status} totalLatency={metadata?.total_latency} />
        </div>

        {(metrics || metadata || status !== 'idle') && (
          <div className="mt-8">
            <TechnicalDetails metrics={metrics} metadata={metadata} />
          </div>
        )}
      </main>

      <Footer />
    </div>
  )
}
