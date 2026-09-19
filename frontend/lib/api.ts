export async function checkBackendHealth(apiBaseUrl: string): Promise<boolean> {
  try {
    const response = await fetch(`${apiBaseUrl}/health`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    })
    return response.ok
  } catch {
    return false
  }
}

export async function checkBackendReady(apiBaseUrl: string): Promise<boolean> {
  try {
    const response = await fetch(`${apiBaseUrl}/ready`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    })
    if (!response.ok) return false
    const data = await response.json()
    return data.ready === true
  } catch {
    return false
  }
}

export function getApiBaseUrl(): string {
  return process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000'
}
