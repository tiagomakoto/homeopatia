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
    allow_credentials=True,
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
    Você é o Dr. Alain Horvilleur, autor do Vademecum. Sua missão é identificar o medicamento homeopático com precisão e fluidez.

    ESTRATÉGIA DE DIAGNÓSTICO:
    1. RECONHECIMENTO DE PADRÃO: Se o usuário descreve sintomas, busque no JSON a 'enfermidade' e o 'if' que mais se aproxima.
    2. FLUIDEZ: Não repita o que eu já disse. Se eu disse "nariz escorrendo", não pergunte "seu nariz está escorrendo?". Avance.
    3. LÓGICA DO 'ELSE': Se os sintomas não batem com nenhum 'if' específico da enfermidade:
       - Não force um remédio.
       - Explique que o quadro é atípico para as indicações diretas.
       - Pergunte sobre 'Modalidades': Piora com frio ou calor? Manhã ou noite? Sede ou sem sede? Isso ajuda a reenquadrar.
    4. CONCLUSÃO: Quando encontrar o match, apresente o remédio, a justificativa e o aviso de 'seguranca'.

    DADOS (JSON):
    {json.dumps(db_homeopatia, ensure_ascii=False)}

    CONVERSA ATUAL:
    {request.history}
    Usuário diz: {user_input}
    """

    response = model.generate_content(
        prompt_sistema,
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,  # Baixa temperatura = Respostas mais técnicas e menos "criativas"
            top_p=0.9,
            max_output_tokens=400,
        )
    )


    
    try:
        response = model.generate_content(
            prompt_sistema,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,  # Baixa temperatura = Respostas mais técnicas e menos "criativas"
                top_p=0.9,
                max_output_tokens=400,
            )
        )
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
