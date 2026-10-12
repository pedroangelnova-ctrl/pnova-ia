import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

# Configuración de CORS total para evitar cualquier bloqueo con el navegador
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- CONFIGURACIÓN REAL DE PNOVA -----------------
HF_TOKEN = "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"  
USER_HF = "pnovagroup"  
# ---------------------------------------------------------------

# URL oficial corregida de la API de Inferencia de Hugging Face
API_URL = f"https://huggingface.co{USER_HF}/PNova-1.5"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

class ChatMessage(BaseModel):
    role: str
    content: str

class OpenAICompatibleRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    stream: Optional[bool] = False

# Función centralizada para procesar los datos de Hugging Face sin romper el servidor
def procesar_peticion_hf(request: OpenAICompatibleRequest):
    ultimo_mensaje = request.messages[-1].content if request.messages else ""
    
    payload = {
        "inputs": f"System: Eres PNova-1.5, una IA avanzada.\nUser: {ultimo_mensaje}\nAssistant:",
        "parameters": {
            "max_new_tokens": 250, 
            "temperature": 0.7,
            "return_full_text": False
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        output = response.json()
        
        # Log de seguridad para la consola de Render
        print(f"HUGGING FACE RESPONSE: {output}")
        
        # Caso 1: El modelo está cargándose en los servidores de Hugging Face
        if isinstance(output, dict) and "estimated_time" in output:
            texto_ia = "Me estoy despertando en los servidores de Hugging Face... Por favor, vuelve a enviar tu mensaje en 20 segundos."
        
        # Caso 2: Error controlado de la API externa
        elif isinstance(output, dict) and "error" in output:
            raise HTTPException(status_code=503, detail=output["error"])
            
        # Caso 3: Formato Estándar -> Lista con diccionario interno [ { "generated_text": "..." } ]
        elif isinstance(output, list) and len(output) > 0:
            primer_elemento = output[0]
            if isinstance(primer_elemento, dict):
                texto_ia = primer_elemento.get("generated_text", "").strip()
            else:
                texto_ia = str(primer_elemento).strip()
            
        # Caso 4: Caída por si viene texto plano directo
        elif isinstance(output, str):
            texto_ia = output.strip()
            
        else:
            texto_ia = "Lo siento, el motor de PNova no ha devuelto un formato reconocible en este momento."

        # Quitamos referencias duplicadas si el modelo repite etiquetas del prompt
        texto_ia = texto_ia.replace("Assistant:", "").strip()

        # Respuesta exacta en el formato OpenAI (choices.message.content)
        return {
            "id": "chatcmpl-pnova",
            "object": "chat.completion",
            "model": "PNova-1.5",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": texto_ia
                    },
                    "finish_reason": "stop"
                }
            ]
        }
        
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Ruta A: Por si tu JavaScript apunta a /chat
@app.post("/chat")
async def chat_directo(request: OpenAICompatibleRequest):
    return procesar_peticion_hf(request)

# Ruta B: Por si tu JavaScript busca el formato oficial de OpenAI /v1/chat/completions
@app.post("/v1/chat/completions")
async def chat_openai(request: OpenAICompatibleRequest):
    return procesar_peticion_hf(request)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
