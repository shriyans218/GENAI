import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq

app = FastAPI(title="SmartSummariser", description="AI Text Summarisation Agent")

client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

SYSTEM_PROMPT = """You are a professional text summarisation agent.
When given any text, respond ONLY with a valid JSON object in this exact format (no markdown, no backticks, no explanation):
{
  "headline": "One-line headline summarising the text",
  "summary": "A 3-sentence summary of the text.",
  "key_points": [
    "Key point 1",
    "Key point 2",
    "Key point 3"
  ]
}"""

class SummariseRequest(BaseModel):
    text: str

@app.post("/summarise")
async def summarise(request: SummariseRequest):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request.text}
        ],
        temperature=0.3,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except Exception:
        return {"raw_response": raw}

@app.get("/")
def health():
    return {"status": "SmartSummariser is running", "model": "GEMINI"}
