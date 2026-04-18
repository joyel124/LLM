from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx, os, time

load_dotenv()

API_KEY    = os.getenv("API_KEY")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL      = os.getenv("MODEL", "gemma4:latest")

app = FastAPI(title="Gemma API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Autenticación por API Key ──────────────────────────
def verificar_api_key(request: Request):
    key = request.headers.get("X-API-Key")
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="API Key inválida")

# ── Modelos de datos ───────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    system: str = "Eres un asistente útil que responde en español."
    history: list = []   # lista de {"role": "user/assistant", "content": "..."}

class ChatResponse(BaseModel):
    response: str
    model: str
    time_seconds: float

# ── Endpoints ──────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "model": MODEL, "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verificar_api_key)])
async def chat(req: ChatRequest):
    # Construir prompt con historial
    prompt = f"System: {req.system}\n\n"
    for msg in req.history:
        rol = "Usuario" if msg["role"] == "user" else "Asistente"
        prompt += f"{rol}: {msg['content']}\n"
    prompt += f"Usuario: {req.message}\nAsistente:"

    start = time.time()

    async with httpx.AsyncClient(timeout=120) as client:
        try:
            res = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": MODEL, "prompt": prompt, "stream": False}
            )
            res.raise_for_status()
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="Ollama no está corriendo")

    data = res.json()
    return ChatResponse(
        response=data["response"].strip(),
        model=MODEL,
        time_seconds=round(time.time() - start, 2)
    )

@app.get("/models", dependencies=[Depends(verificar_api_key)])
async def listar_modelos():
    async with httpx.AsyncClient() as client:
        res = await client.get(f"{OLLAMA_URL}/api/tags")
    return res.json()