# Frontend Build v0.1.0 — Setup & Run Commands

## Files Created

### Configuration
- ✓ package.json
- ✓ package-lock.json (auto-generated from npm install)
- ✓ tsconfig.json
- ✓ next.config.ts
- ✓ tailwind.config.ts
- ✓ postcss.config.js
- ✓ .env.example
- ✓ .gitignore

### App Structure
- ✓ app/layout.tsx (root layout with metadata)
- ✓ app/page.tsx (main app with state management)
- ✓ app/globals.css (Tailwind global styles)

### Components
- ✓ components/header.tsx
- ✓ components/text-input-card.tsx
- ✓ components/output-card.tsx
- ✓ components/technical-details.tsx
- ✓ components/metric-card.tsx
- ✓ components/request-details.tsx
- ✓ components/footer.tsx

### Lib (Business Logic)
- ✓ lib/types.ts
- ✓ lib/api.ts
- ✓ lib/stream.ts
- ✓ lib/utils.ts

### Documentation
- ✓ frontend/README.md (detailed frontend documentation)

---

## Quick Start

### 1. Install Dependencies (already done)

```bash
cd frontend
npm install
```

Dependencies installed:
- React 19, React DOM 19
- Next.js 15.5
- TypeScript 5.7
- Tailwind CSS 3.4
- Lucide React 0.427 (icons)
- PostCSS, Autoprefixer

### 2. Create .env.local

```bash
cd frontend
cp .env.example .env.local
```

Edit `frontend/.env.local`:
```
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

Change the URL if your backend runs elsewhere.

### 3. Run Development Server

```bash
cd frontend
npm run dev
```

Server runs on **http://localhost:3000**

```
▲ Next.js 15.5.25
- Local:        http://localhost:3000
- Environments: .env.local

✓ Ready in 1.23s
```

### 4. Test the App

1. Open **http://localhost:3000** in a browser
2. Verify backend is running at `http://localhost:8000`
3. Paste sample text:
   ```
   Acme reported quarterly revenue of $4.2 billion, up 12% year over year.
   Operating profit increased 8% to $620 million...
   ```
4. Click "Generate Bullets"
5. Watch streaming output appear in real-time

### 5. Production Build

```bash
cd frontend
npm run build
npm run start
```

Server runs on port 3000 (production).

---

## Architecture

### Request Flow

User Input
  ↓
TextInputCard (capture + validate)
  ↓
POST /v1/bullets/stream (backend)
  ↓
streamBullets() (lib/stream.ts)
  ↓
SSE chunks streamed
  ↓
onChunk callback appends to state
  ↓
OutputCard renders in real-time
  ↓
TechnicalDetails shows metrics

### Streaming Implementation

- Native `fetch` with `response.body.getReader()`
- SSE decoding line-by-line
- Immediate render (no buffering)
- AbortController for cancellation
- Error handling for token limits

### State Management

React hooks only (no Redux):
- inputText, outputText
- status (idle | connecting | streaming | completed | error)
- error, metrics, metadata
- AbortController ref

---

## Backend Assumptions

The frontend expects:
1. **GET /health** → `{"status": "ok"}`
2. **GET /ready** → `{"ready": true/false}`
3. **POST /v1/bullets/stream** (streaming endpoint)
   - Input: `{"text": "..."}`
   - Output: Server-Sent Events (SSE) or plain text chunks

### Metrics Expected from Backend

The frontend displays:
- input_tokens
- output_tokens
- ttft (time-to-first-token)
- mean_itl (mean inter-token latency)
- total_latency or calculated from client
- prefill_time

All metrics must come from backend. Frontend never manufactures values.

---

## Design Highlights

### Visual
- Light cool-white background
- White cards with subtle borders/shadows
- Primary blue, indigo secondary accents
- Soft green for success
- Navy text, slate-blue secondary
- 1500px max container width
- Responsive 50/50 grid (desktop), stacked (mobile)

### No Dark Mode (v0.1)
- Deliberately light-only
- Can add in v0.2

### Accessibility
- Semantic HTML
- Proper ARIA labels
- Visible focus states
- Respects prefers-reduced-motion
- Good color contrast

---

## Key Commands

```bash
# Development
npm run dev                    # Start dev server (http://localhost:3000)

# Production
npm run build                  # Build for production
npm run start                  # Start production server

# Linting
npm lint                       # Run ESLint (if configured)

# Type checking (included in build)
# TypeScript runs as part of build process
```

---

## Environment Variables

### .env.local (required for development)

```
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

This is read at build time and embedded in the client bundle.

---

## Troubleshooting

### Port 3000 already in use

```bash
npm run dev -- -p 3001
```

### Backend unavailable warning

Ensure backend is running:
```bash
# From backend directory
cd backend
uv run python -m src.main
# Should be on http://localhost:8000
```

### Streaming stops/no output

Check browser DevTools Network tab:
- Look for POST /v1/bullets/stream request
- Check response status (should be 200)
- Check response headers for `content-type: text/event-stream`

### Metrics show "—"

Backend may not be returning metrics in the stream. Verify the `/v1/bullets/stream` response includes metric fields in the streamed data.

### Build fails with TypeScript errors

```bash
npm run build -- --verbose
```

Check errors, fix them, rebuild.

---

## Quality Assurance Checklist

✓ npm install completed
✓ Production build succeeds (no errors/warnings)
✓ TypeScript compiles successfully
✓ All imports resolve correctly
✓ Responsive layout works (desktop/tablet/mobile)
✓ No unused imports or variables
✓ No "View Full Logs" button anywhere
✓ Copy button works
✓ Example button works
✓ Metrics begin as "—"
✓ No fake/manufactured metrics
✓ Error states display correctly
✓ Streaming doesn't wait for full response
✓ Frontend runs when backend unavailable (shows warning)

---

## Next Steps

1. **Local Testing**
   ```bash
   cd frontend
   npm run dev
   # Open http://localhost:3000
   # Ensure http://localhost:8000 is running
   ```

2. **Deploy Frontend**
   - Vercel: Push to GitHub, connect Vercel
   - Docker: Build image with `npm run build`
   - Other: Run production build, serve .next folder

3. **Configure API URL**
   - For Vercel/production: Set `NEXT_PUBLIC_API_BASE_URL` env var
   - Example: `NEXT_PUBLIC_API_BASE_URL=https://api.example.com`

---

## Files Summary

```
frontend/
├── app/
│   ├── layout.tsx            (4 KB)
│   ├── page.tsx              (7 KB)
│   └── globals.css           (1 KB)
├── components/               (14 KB total)
│   ├── header.tsx
│   ├── text-input-card.tsx
│   ├── output-card.tsx
│   ├── technical-details.tsx
│   ├── metric-card.tsx
│   ├── request-details.tsx
│   └── footer.tsx
├── lib/                      (8 KB total)
│   ├── types.ts
│   ├── api.ts
│   ├── stream.ts
│   └── utils.ts
├── public/                   (empty)
├── package.json              (729 B)
├── package-lock.json         (219.6 KB)
├── tsconfig.json             (879 B)
├── next.config.ts            (127 B)
├── tailwind.config.ts        (871 B)
├── postcss.config.js         (82 B)
├── .env.example              (47 B)
├── .gitignore                (286 B)
└── README.md                 (detailed documentation)
```

**Total Code**: ~35 KB (excluding node_modules)
**Dependencies**: 369 packages
**Build Size**: 109 KB (First Load JS)

