import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Agro-Coach Mobile", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 🔑 CLÉ API (À MODIFIER 1 FOIS)
# Pour Groq, utilise une clé commençant par gsk_
# Pour OpenAI, sk-...
GROQ_KEY = "gsk_TaCléGroqIciSansEspaces"  # ← Remplace par ta vraie clé

# Initialisation explicite (ignore les variables système)
llm = None
if GROQ_KEY and GROQ_KEY.startswith("gsk_"):
    try:
        llm = ChatOpenAI(
            model="qwen-2.5-32b",
            api_key=GROQ_KEY,  # ✅ Passé explicitement
            base_url="https://api.groq.com/openai/v1",
            temperature=0.1,
            max_tokens=800
        )
        logger.info("✅ LLM initialisé avec Groq + Qwen")
    except Exception as e:
        logger.error(f"❌ Erreur LLM: {e}")
else:
    logger.warning("⚠️ GROQ_KEY non configurée ou invalide")

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    location: str = Field(default="Banakorosso")
    user_id: str = Field(default="anonymous")

@app.get("/")
def home(): return {"service": "Agro-Coach", "status": "running" if llm else "degraded"}

@app.get("/health")
def health():
    return {"status": "healthy", "model": "qwen-2.5-32b"} if llm else {"status": "degraded", "missing": "GROQ_KEY"}

@app.post("/ask")
async def ask(req: QueryRequest):
    if not llm:
        return {"status": "error", "reponse": "Clé API manquante", "conseils": []}
    try:
        prompt = f"""Lieu: {req.location}. Question: {req.question}.
Réponds avec EXACTEMENT 3 conseils courts numérotés, <200 mots, chiffres concrets (FCFA/litres), français simple, contexte Burkina Faso.
Format:
1. [conseil]
2. [conseil]
3. [conseil]"""
        resp = llm.invoke(prompt)
        texte = resp.content.strip()
        lignes = [l.strip().lstrip("0123456789.-) ") for l in texte.split('\n') if l.strip() and not l.startswith('```')]
        return {"status": "success", "reponse": texte, "conseils": lignes[:3]}
    except Exception as e:
        return {"status": "error", "reponse": f"Erreur: {str(e)[:150]}", "conseils": []}
