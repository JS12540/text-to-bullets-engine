# Text-to-Bullets Frontend

A production-ready Next.js application for generating bullet points from text using a fine-tuned T5 model.

## Overview

This frontend provides a polished, technical UI for the text-to-bullets inference backend. Users can paste long-form text and receive concise, actionable bullet points via real-time streaming.

## Tech Stack

- **Framework**: Next.js 15 with App Router
- **Language**: TypeScript 5.7
- **Styling**: Tailwind CSS 3.4
- **Icons**: Lucide React 0.376
- **HTTP**: Native fetch API
- **Streaming**: Server-Sent Events (SSE)

## Project Structure

```
frontend/
├── app/
│   ├── layout.tsx          # Root layout with metadata
│   ├── page.tsx            # Main page with state management
│   └── globals.css         # Global Tailwind styles
├── components/
│   ├── header.tsx          # Title, feature badges, AI badge
│   ├── text-input-card.tsx # Input textarea, example, advanced options
│   ├── output-card.tsx     # Streamed output, copy button
│   ├── technical-details.tsx # Metrics dashboard
│   ├── metric-card.tsx     # Single metric display
│   ├── request-details.tsx # Collapsible request metadata
│   └── footer.tsx          # Footer with copyright
├── lib/
│   ├── types.ts            # TypeScript interfaces
│   ├── api.ts              # Backend health/readiness checks
│   ├── stream.ts           # SSE streaming logic
│   └── utils.ts            # Format and utility functions
├── public/                 # Static assets (empty in v0.1)
├── package.json
├── tsconfig.json
├── next.config.ts
├── tailwind.config.ts
├── postcss.config.js
├── .env.example
└── README.md
```

## Environment Setup

### Install Dependencies

```bash
cd frontend
npm install
```

### Configure Backend URL

Create `.env.local`:

```bash
cp .env.example .env.local
```

Edit `.env.local`:

```
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

Change the URL if your backend runs elsewhere.

## Local Development

```bash
cd frontend
npm run dev
```

Frontend: `http://localhost:3000`

Backend must be running at the configured `NEXT_PUBLIC_API_BASE_URL` (default `http://localhost:8000`).

## Streaming Architecture

### Request Flow

1. User enters text and clicks "Generate Bullets"
2. Fetch POST to `/v1/bullets/stream` with JSON body `{ text: "..." }`
3. Backend streams chunks via SSE (Server-Sent Events)
4. Frontend appends chunks to output in real-time
5. No waiting for full response; renders as chunks arrive

### Streaming Implementation

**File**: `lib/stream.ts`

- Uses native `fetch` with `response.body.getReader()`
- Decodes SSE events line-by-line
- Handles both text and JSON event formats
- Supports `AbortController` for cancellation
- Detects token-limit errors from backend

### Expected Backend Format

The backend streams plain text or JSON chunks. The frontend handles:

```
data: "- Machine learning\n"
data: "- Transforms AI\n"
```

For metrics (if returned):

```
data: {"metrics": {"input_tokens": 112, "ttft": 0.68, ...}}
```

## Component Responsibilities

### `Header`
- Title, description, feature badges (Fast, Private, CPU Optimized)
- AI Powered badge
- Responsive on mobile

### `TextInputCard`
- Large textarea with 10,000 character limit (frontend guard)
- Example button to populate sample text
- Advanced options (collapsible, read-only for v0.1)
- Generate button with loading states

### `OutputCard`
- Displays streamed bullet points in real-time
- Shows green "Completed" badge with duration when done
- Copy button (temporary "Copied" state)
- Empty state before generation

### `TechnicalDetails`
- Six metric cards: Total Latency, TTFT, Mean ITL, Input Tokens, Output Tokens, Model
- Real metrics from backend (never manufactured)
- Shows "—" for unavailable values
- Expandable Request Details section below

### `MetricCard`
- Icon, label, value for a single metric
- Used by TechnicalDetails and RequestDetails

### `RequestDetails`
- Collapsible section inside TechnicalDetails
- Displays metadata: request ID, status, all latencies, token counts, EOS flag
- Shows "—" for missing fields

### `Footer`
- Logo, title, tagline
- Built for learning and real-world inference message

## Styling

### Design Language

- **Background**: Very light cool-white (`#f9fafb`)
- **Cards**: White with subtle borders and shadows
- **Primary**: Vivid blue (`#2563eb`)
- **Secondary**: Indigo (`#4f46e5`)
- **Text**: Navy (`#1e293b`), Slate-blue (`#64748b`)
- **Success**: Soft green (`#10b981`)
- **Spacing**: Generous, responsive

### Responsive Design

- **Desktop**: Input and Output side-by-side (50/50 grid)
- **Tablet/Mobile**: Stacked vertically
- **Header**: Badges reflow on small screens
- **Metrics**: 3-column grid on medium, 1-2 on mobile

### Accessibility

- Semantic HTML headings
- Button labels and aria attributes
- Visible focus states
- Sufficient color contrast
- Respects `prefers-reduced-motion`

## Error Handling

Displays inline error cards for:

- Backend unavailable
- Request timeout
- Context length exceeded (token limit)
- Network errors
- Unexpected server errors

No browser `alert()` dialogs. Errors appear below the header.

## State Management

Uses React hooks only (no Redux/context):

- `inputText` — User input
- `outputText` — Generated bullets
- `status` — idle | connecting | streaming | completed | error
- `error` — Error message string or null
- `metrics` — Backend metrics
- `metadata` — Request metadata
- `backendReady` — Health check result

`AbortController` ref for request cancellation.

## Production Build

```bash
npm run build
npm run start
```

Server runs on port 3000.

## Metrics Display

All metrics come from the backend:

- **Total Latency**: Complete request duration (ms)
- **TTFT**: Time to first generated token (ms)
- **Mean ITL**: Average latency per decode step (ms)
- **Input Tokens**: Tokenized system + task + user text
- **Output Tokens**: Count of generated tokens
- **Model**: T5 INT8
- **Quantization**: TorchAO

Before first request, all numeric values show "—".

No fake metrics are ever displayed.

## API Configuration

Backend URL is read from `NEXT_PUBLIC_API_BASE_URL` env var at build time.

Example:

```bash
NEXT_PUBLIC_API_BASE_URL=https://api.example.com npm run build
```

## Key Assumptions About Backend

1. Exposes `/health` (returns `{"status": "ok"}`)
2. Exposes `/ready` (returns `{"ready": true/false}`)
3. Exposes `POST /v1/bullets/stream` streaming endpoint
4. Request body: `{"text": "..."}`
5. Streams plain text or JSON events
6. Returns 400 with token-limit error on input > max
7. Metrics included in streamed response or final response
8. Greedy decoding (no beam search options in v0.1)

## Development Tips

### Check Backend Connection

The app checks backend readiness on mount. If unavailable, a warning appears below the header.

### Test Streaming Locally

```bash
# Terminal 1: Backend
cd backend
uv run python -m src.main

# Terminal 2: Frontend
cd frontend
npm run dev

# Terminal 3: Test with curl
curl -X POST http://localhost:8000/v1/bullets/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "Your test text"}'
```

### Debug Metrics

Open browser DevTools Network tab. Stream requests should show status 200 and proper SSE headers.

## v0.1.0 Scope

**Included:**
- ✅ Streaming input/output UI
- ✅ Real backend metrics
- ✅ Error handling
- ✅ Example button
- ✅ Copy to clipboard
- ✅ Technical details dashboard
- ✅ Responsive design
- ✅ Accessibility basics

**Not included (v0.2+):**
- ❌ Dark mode
- ❌ Beam search options
- ❌ Model selection
- ❌ History/saved outputs
- ❌ Authentication
- ❌ Rate limiting UI

## Troubleshooting

### "Backend server unavailable"

Ensure backend is running and `NEXT_PUBLIC_API_BASE_URL` is correct.

```bash
# Test backend
curl http://localhost:8000/health
```

### Streaming stops partway

Check network tab for aborted requests. Verify backend is not timing out.

### Metrics show "—"

Backend may not be returning metrics. Verify the `/v1/bullets/stream` response includes metric fields.

### Copy button not working

Check browser console for clipboard API errors. Some contexts (http://localhost) may require HTTPS.
