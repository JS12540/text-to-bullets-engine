"""Vercel Python entrypoint. Vercel's Python runtime looks for an ASGI `app`
object in this file and serves it directly — no uvicorn needed in production.

Locally, run the server with uvicorn instead (see README.md); this file is
only exercised by `vercel dev` / the deployed function.
"""
from src.main import create_app

app = create_app()
