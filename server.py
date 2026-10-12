import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Configuración total de CORS para evitar bloqueos con tu dominio
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

# URL oficial de la API de Inferencia de Hugging Face
API_URL = f"https://huggingface.co{USER_HF}/PNova-1.5"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

class ChatMessage(BaseModel):
    role: str
    content: str

class OpenAICompatibleRequest(BaseModel):
    model: str = None
    messages: List[ChatMessage]
    stream: bool = False

@app.post("/chat")
async def chat_endpoint(request: OpenAICompatibleRequest):
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
        
        # Log de depuración en la consola de Render para ver qué responde la API externa
        print(f"RESPUESTA DE HUGGING FACE: {output}")
        
        # Caso A: Si el modelo está dormido en Hugging Face (Evita que el servidor tire un 500)
        if isinstance(output, dict) and "estimated_time" in output:
            texto_ia = "Me estoy despertando en los servidores de Hugging Face... Por favor, vuelve a enviar tu mensaje en 20 segundos."
        
        # Caso B: Formato correcto -> Lista con diccionario interno [ { generated_text: ... } ]
        elif isinstance(output, list) and len(output) > 0:
            # Accedemos estrictamente al elemento 0 de la lista (que es el diccionario)
            primer_elemento = output[0]
            texto_ia = primer_elemento.get("generated_text", "").strip()
            
        # Caso C: Error controlado enviado por la API
        elif isinstance(output, dict) and "error" in output:
            raise HTTPException(status_code=503, detail=output["error"])
        else:
            texto_ia = "Lo siento, el motor de PNova no ha devuelto un formato reconocible en este momento."

        # Retornamos el JSON EXACTO con la estructura OpenAI (choices.message.content)
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
        print(f"ERROR EN EL ENDPOINT: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
