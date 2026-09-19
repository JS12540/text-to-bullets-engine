import { GenerationMetrics, TokenLimitError } from './types'

export interface StreamResult {
  text: string
  metrics: GenerationMetrics
}

// Backend reports durations in seconds; the UI formats everything in ms.
function toMs(metrics: GenerationMetrics): GenerationMetrics {
  return {
    ...metrics,
    total_latency: metrics.total_latency !== undefined ? metrics.total_latency * 1000 : undefined,
    ttft: metrics.ttft !== undefined ? metrics.ttft * 1000 : undefined,
    mean_itl: metrics.mean_itl !== undefined ? metrics.mean_itl * 1000 : undefined,
    prefill_time: metrics.prefill_time !== undefined ? metrics.prefill_time * 1000 : undefined,
  }
}

export async function streamBullets(
  text: string,
  apiBaseUrl: string,
  onChunk: (chunk: string) => void,
  onMetrics: (metrics: GenerationMetrics) => void,
  signal?: AbortSignal
): Promise<StreamResult> {
  const controller = new AbortController()
  const finalSignal = signal || controller.signal

  let fullText = ''
  let metrics: GenerationMetrics = {}

  try {
    const response = await fetch(`${apiBaseUrl}/v1/bullets/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
      signal: finalSignal,
    })

    if (!response.ok) {
      // Handle error responses
      const contentType = response.headers.get('content-type')
      if (contentType?.includes('application/json')) {
        const error = await response.json()

        // Check for token limit error
        if (error.detail?.error_code === 'TOKEN_LIMIT_EXCEEDED') {
          const tokenError = error.detail as TokenLimitError
          throw new Error(
            `Input exceeds maximum length. Input: ${tokenError.actual_tokens} tokens, Max: ${tokenError.max_allowed_tokens} tokens`
          )
        }

        throw new Error(error.detail?.message || 'Failed to generate bullets')
      }
      throw new Error(`Server error: ${response.status}`)
    }

    // Handle streaming response
    if (!response.body) {
      throw new Error('No response body')
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')

      // Keep the last incomplete line in the buffer
      buffer = lines[lines.length - 1]

      for (let i = 0; i < lines.length - 1; i++) {
        const line = lines[i].trim()

        if (line.startsWith('data: ')) {
          const data = line.slice(6)

          // Try to parse as JSON for metrics
          try {
            const parsed = JSON.parse(data)
            if (parsed.metrics) {
              metrics = toMs(parsed.metrics)
              onMetrics(metrics)
            } else if (parsed.text) {
              fullText += parsed.text
              onChunk(parsed.text)
            }
          } catch {
            // Not JSON, treat as plain text chunk
            fullText += data
            onChunk(data)
          }
        }
      }
    }

    // Process any remaining buffer
    if (buffer.trim().startsWith('data: ')) {
      const data = buffer.trim().slice(6)
      try {
        const parsed = JSON.parse(data)
        if (parsed.metrics) {
          metrics = toMs(parsed.metrics)
          onMetrics(metrics)
        } else if (parsed.text) {
          fullText += parsed.text
          onChunk(parsed.text)
        }
      } catch {
        fullText += data
        onChunk(data)
      }
    }

    return { text: fullText, metrics }
  } catch (error) {
    if (error instanceof Error) {
      if (error.name === 'AbortError') {
        throw new Error('Request cancelled')
      }
      throw error
    }
    throw new Error('Unknown error occurred')
  }
}
