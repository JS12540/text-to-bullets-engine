"""System and task prompts. Literal copy of backend/src/constants/prompts.py."""

SYSTEM_PROMPT = """You are an expert at converting text into concise, actionable bullet points.
Transform the input text into a clear list of key points using the following format:
- Each point should be a complete, standalone thought
- Use simple, direct language
- Remove redundancy
- Focus on the most important information
- Mark each point with <BULLET>"""

TASK_PROMPT = "Convert the following text into bullet points:\n"
