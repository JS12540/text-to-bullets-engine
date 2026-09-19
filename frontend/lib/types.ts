export interface BulletsRequest {
  text: string
  temperature?: number
}

export interface GenerationMetrics {
  input_tokens?: number
  output_tokens?: number
  prefill_time?: number
  ttft?: number
  mean_itl?: number
  total_latency?: number
  reached_eos?: boolean
}

export interface RequestMetadata {
  request_id?: string
  status?: string
  input_tokens?: number
  output_tokens?: number
  prefill_latency?: number
  ttft?: number
  mean_itl?: number
  total_latency?: number
  reached_eos?: boolean
  model?: string
  quantization?: string
}

export type GenerationStatus = 'idle' | 'connecting' | 'streaming' | 'completed' | 'error'

export interface TokenLimitError {
  error_code?: string
  message?: string
  actual_tokens?: number
  max_allowed_tokens?: number
}
