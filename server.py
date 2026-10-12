import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CONFIGURACIÓN REAL DE PNOVA
HF_TOKEN = "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"  
USER_HF = "pnovagroup"  

# Endpoint oficial de inferencia para servidores en la nube
API_URL = f"https://api-inference.huggingface.co/models/{USER_HF}/PNova-1.5"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    payload = {
        "inputs": f"System: Eres PNova-1.5, una IA avanzada.\nUser: {request.message}\nAssistant:",
        "parameters": {
            "max_new_tokens": 250, 
            "temperature": 0.7,
            "return_full_text": False
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        output = response.json()
        
        if isinstance(output, dict) and "estimated_time" in output:
            return {"response": "Me estoy despertando en los servidores de Hugging Face... Vuelve a enviar el mensaje en 20 segundos."}
            
        if isinstance(output, list) and len(output) > 0:
            respuesta_ia = output[0].get("generated_text", "").strip()
            return {"response": respuesta_ia}
        elif isinstance(output, dict) and "error" in output:
            raise HTTPException(status_code=503, detail=output["error"])
        
        return {"response": "Lo siento, el motor de PNova no ha devuelto un formato válido."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
