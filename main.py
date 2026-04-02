import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path


# Define a base como o diretório onde o main.py está
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "624624112-Homeopatia.json"

if not DATA_FILE.exists():
    print(f"Erro Crítico: O arquivo {DATA_FILE} não foi encontrado no repositório.")

# SEGURANÇA: O código busca a chave nas configurações do servidor
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Erro: A variável de ambiente GOOGLE_API_KEY não foi configurada.")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

app = FastAPI()

# Permitir que seu site na HostGator acesse o backend no Render
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Depois você pode trocar pelo seu domínio oficial
    allow_methods=["*"],
    allow_headers=["*"],
)


# Carregar o banco de dados JSON (o arquivo que você me enviou)
def load_db():
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

db_homeopatia = load_db()

class ChatRequest(BaseModel):
    message: str
    history: list = [] # Para manter o contexto da conversa

@app.post("/ask")
async def ask_homeopatia(request: ChatRequest):
    user_input = request.message
    
    # PROMPT DE SISTEMA: Define como a IA deve se comportar usando o seu JSON
    prompt_sistema = f"""
    Você é um assistente especializado em Homeopatia, baseado estritamente no Vademecum de Alain Horvilleur.
    Seu objetivo é guiar o usuário até o medicamento correto usando os dados fornecidos abaixo.

    REGRAS DE OURO:
    1. Use o campo 'entrada_primaria' para confirmar a enfermidade.
    2. Use o 'diagnostico_diferencial' para fazer perguntas de 'SIM ou NÃO' sobre os sintomas (campos 'if').
    3. Nunca invente medicamentos fora do JSON.
    4. Sempre inclua o texto de 'seguranca' da enfermidade correspondente.
    5. Se o usuário for vago (ex: "dor"), pergunte a localização com base nas enfermidades disponíveis.

    DADOS DO VADEMECUM (JSON):
    {json.dumps(db_homeopatia, ensure_ascii=False)}

    HISTÓRICO DA CONVERSA:
    {request.history}

    ENTRADA DO USUÁRIO:
    {user_input}
    """

    try:
        response = model.generate_content(prompt_sistema)
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
