import os
from fastapi import FastAPI
from pydantic import BaseModel
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

app = FastAPI()

# --- Agent Definition ---
SYSTEM_PROMPT = """
You are a professional text summarisation agent.
When given any text, respond ONLY with a valid JSON object in this exact format:
{
  "headline": "One-line headline summarising the text",
  "summary": "A 3-sentence summary of the text.",
  "key_points": [
    "Key point 1",
    "Key point 2",
    "Key point 3"
  ]
}
Do not include any explanation or markdown. Return raw JSON only.
"""

agent = LlmAgent(
    name="smartsummariser",
    model="gemini-1.5-flash",
    instruction=SYSTEM_PROMPT,
)

session_service = InMemorySessionService()
runner = Runner(agent=agent, app_name="smartsummariser", session_service=session_service)

# --- Request/Response Schema ---
class SummariseRequest(BaseModel):
    text: str

# --- Endpoint ---
@app.post("/summarise")
async def summarise(request: SummariseRequest):
    session = await session_service.create_session(
        app_name="smartsummariser",
        user_id="user",
    )
    message = types.Content(
        role="user",
        parts=[types.Part(text=request.text)]
    )
    result_text = ""
    async for event in runner.run_async(
        user_id="user",
        session_id=session.id,
        new_message=message,
    ):
        if event.is_final_response() and event.content:
            for part in event.content.parts:
                if part.text:
                    result_text += part.text

    import json
    try:
        return json.loads(result_text)
    except Exception:
        return {"raw_response": result_text}

@app.get("/")
def health():
    return {"status": "SmartSummariser is running"}
