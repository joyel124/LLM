from fastapi import FastAPI, HTTPException, Depends, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from datetime import datetime
import os, time, shutil, httpx

load_dotenv()

# ── Config ─────────────────────────────────────────────
API_KEY    = os.getenv("API_KEY", "clave-secreta-123")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL      = os.getenv("MODEL", "gemma4:e2b")
DOCS_DIR   = "./documentos"
CHROMA_DIR = "./chroma_db"

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

app = FastAPI(title="Gemma API con RAG + Agente", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── LangChain imports (compatibles con versión actual) ──
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun

# ── LLM y Embeddings ───────────────────────────────────
llm        = OllamaLLM(model=MODEL, base_url=OLLAMA_URL, temperature=0.7)
embeddings = OllamaEmbeddings(model=MODEL, base_url=OLLAMA_URL)

# ── RAG: Vector Store ──────────────────────────────────
vectorstore = None
retriever   = None

def cargar_documentos():
    global vectorstore, retriever

    archivos = [f for f in os.listdir(DOCS_DIR) if f.endswith((".pdf", ".txt"))]
    if not archivos:
        print("⚠️  Sin documentos — RAG desactivado")
        vectorstore = None
        retriever   = None
        return

    print(f"📚 Indexando {len(archivos)} documento(s)...")
    docs = []
    for archivo in archivos:
        ruta = os.path.join(DOCS_DIR, archivo)
        try:
            loader = PyPDFLoader(ruta) if archivo.endswith(".pdf") else TextLoader(ruta, encoding="utf-8")
            docs.extend(loader.load())
        except Exception as e:
            print(f"  ⚠️ Error con {archivo}: {e}")

    if not docs:
        print("⚠️  Ningún documento válido")
        return

    splitter   = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    fragmentos = splitter.split_documents(docs)
    print(f"  ✂️  {len(fragmentos)} fragmentos")

    vectorstore = Chroma.from_documents(
        documents=fragmentos,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print("  ✅ RAG listo")

cargar_documentos()

# ── Herramientas del agente ────────────────────────────
def buscar_en_documentos(query: str) -> str:
    if retriever is None:
        return "No hay documentos cargados. Sube archivos PDF o TXT usando /upload."
    docs = retriever.invoke(query)
    if not docs:
        return "No encontré información relevante en los documentos."
    return "\n\n".join(
        f"[Fuente: {d.metadata.get('source', '?')}]\n{d.page_content}"
        for d in docs
    )

search_tool = DuckDuckGoSearchRun()

herramientas = [
    Tool(
        name="buscar_internet",
        func=search_tool.run,
        description="Busca información actualizada en internet. Úsalo para noticias, precios, eventos recientes o cualquier dato que pueda haber cambiado recientemente."
    ),
    Tool(
        name="buscar_documentos",
        func=buscar_en_documentos,
        description="Busca en los documentos propios del usuario (PDFs y TXTs subidos). Úsalo para preguntas sobre documentos específicos del usuario."
    ),
    Tool(
        name="fecha_hora_actual",
        func=lambda _: datetime.now().strftime("Hoy es %A %d de %B de %Y, son las %H:%M hrs"),
        description="Devuelve la fecha y hora actual exacta. Úsalo cuando pregunten qué día es, la hora, o el año actual."
    ),
]

# ── Prompt ReAct ───────────────────────────────────────
REACT_PROMPT = PromptTemplate.from_template("""Eres un asistente inteligente que responde siempre en español.

Tienes acceso a herramientas para obtener información actualizada. Úsalas cuando:
- Te pregunten algo que puede haber cambiado (noticias, precios, clima, eventos)
- Te pregunten por documentos del usuario
- Te pregunten la fecha o la hora actual
- No estés seguro con tu conocimiento base

Herramientas disponibles:
{tools}

Formato OBLIGATORIO — debes seguirlo exactamente:
Thought: [razona qué necesitas hacer]
Action: [nombre exacto de la herramienta de esta lista: {tool_names}]
Action Input: [texto de entrada para la herramienta]
Observation: [resultado de la herramienta]
... (puedes repetir Thought/Action/Observation si necesitas más pasos)
Thought: Tengo suficiente información para responder
Final Answer: [respuesta final completa en español]

Historial de conversación:
{chat_history}

Pregunta del usuario: {input}
{agent_scratchpad}""")

# ── Agente ─────────────────────────────────────────────
agent = create_react_agent(llm=llm, tools=herramientas, prompt=REACT_PROMPT)
agent_executor = AgentExecutor(
    agent=agent,
    tools=herramientas,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True,
)

# ── Autenticación ──────────────────────────────────────
def verificar_api_key(request: Request):
    key = request.headers.get("X-API-Key")
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="API Key inválida")

# ── Modelos de datos ───────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    history: list = []
    use_agent: bool = True

class ChatResponse(BaseModel):
    response: str
    model: str
    time_seconds: float
    used_tools: bool

# ── Endpoints ──────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "ok",
        "model": MODEL,
        "documentos_indexados": len([f for f in os.listdir(DOCS_DIR) if f.endswith((".pdf",".txt"))]),
        "rag_activo": retriever is not None,
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verificar_api_key)])
async def chat(req: ChatRequest):
    start = time.time()

    # Formatear historial
    chat_history = ""
    for msg in req.history[-6:]:
        rol = "Usuario" if msg["role"] == "user" else "Asistente"
        chat_history += f"{rol}: {msg['content']}\n"

    try:
        if req.use_agent:
            resultado = agent_executor.invoke({
                "input": req.message,
                "chat_history": chat_history
            })
            respuesta  = resultado["output"]
            used_tools = True
        else:
            # Llamada directa a Ollama sin agente
            async with httpx.AsyncClient(timeout=180) as client:
                r = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": MODEL, "prompt": f"{chat_history}Usuario: {req.message}\nAsistente:", "stream": False}
                )
                r.raise_for_status()
            respuesta  = r.json()["response"].strip()
            used_tools = False

    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Ollama no está corriendo")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(
        response=respuesta.strip(),
        model=MODEL,
        time_seconds=round(time.time() - start, 2),
        used_tools=used_tools
    )

@app.post("/upload", dependencies=[Depends(verificar_api_key)])
async def subir_documento(file: UploadFile = File(...)):
    if not file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos PDF y TXT")
    ruta = os.path.join(DOCS_DIR, file.filename)
    with open(ruta, "wb") as f:
        shutil.copyfileobj(file.file, f)
    cargar_documentos()
    return {"mensaje": f"✅ '{file.filename}' subido e indexado correctamente"}

@app.delete("/documentos/{nombre}", dependencies=[Depends(verificar_api_key)])
def eliminar_documento(nombre: str):
    ruta = os.path.join(DOCS_DIR, nombre)
    if not os.path.exists(ruta):
        raise HTTPException(status_code=404, detail="Documento no encontrado")
    os.remove(ruta)
    cargar_documentos()
    return {"mensaje": f"✅ '{nombre}' eliminado y RAG actualizado"}

@app.get("/documentos", dependencies=[Depends(verificar_api_key)])
def listar_documentos():
    archivos = [f for f in os.listdir(DOCS_DIR) if f.endswith((".pdf", ".txt"))]
    return {"documentos": archivos, "total": len(archivos), "rag_activo": retriever is not None}

@app.get("/models", dependencies=[Depends(verificar_api_key)])
async def listar_modelos():
    async with httpx.AsyncClient() as client:
        res = await client.get(f"{OLLAMA_URL}/api/tags")
    return res.json()