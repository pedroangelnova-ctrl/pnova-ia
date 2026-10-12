"""
server.py - Sirve la interfaz (index.html) y tu modelo PNova-1.5 en el mismo puerto.

Uso:
    pip install fastapi uvicorn          (transformers y torch ya los tienes)
    python server.py

Luego abre http://localhost:8000 (en Codespaces: pestaña "Puertos" -> puerto 8000).

Variables opcionales:
    PNOVA_MODEL   carpeta del modelo fusionado (por defecto "PNova-1.5")
    PNOVA_CTX     contexto máximo en tokens   (por defecto 2048; debe coincidir con SERVER_CTX del HTML)
    PORT          puerto                      (por defecto 8000)

Búsqueda en fuentes (PNova 2.0 Pro), se usa el primero que esté configurado:
    SERPER_API_KEY   resultados de Google (serper.dev)
    TAVILY_API_KEY   Tavily (plan gratuito mensual)
    BRAVE_API_KEY    Brave Search API (de pago por uso)
    SEARXNG_URL      tu propio SearXNG, p. ej. http://localhost:8080 (gratis, ilimitado)
    (ninguna)        DuckDuckGo gratis, sin clave (pip install ddgs)
"""
import json
import os
import threading
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

BASE = Path(__file__).resolve().parent
MODEL_DIR = os.environ.get("PNOVA_MODEL", "PNova-1.5")
MODEL_NAME = "PNova-1.5"
MAX_CTX = int(os.environ.get("PNOVA_CTX", "2048"))
PORT = int(os.environ.get("PORT", "8000"))

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Cargando {MODEL_DIR} en {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=dtype).to(device)
model.eval()
print("Modelo listo.")

LOCK = threading.Lock()  # una generación a la vez

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def flatten(content):
    """El HTML puede mandar contenido como lista (texto + imagen). El modelo es solo de texto."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts, has_image = [], False
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                texts.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                has_image = True
        out = "\n".join(t for t in texts if t)
        if has_image:
            out += "\n[El usuario adjuntó una imagen que no puedes ver.]"
        return out
    return str(content or "")


def error(msg, status=400):
    return JSONResponse({"error": {"message": msg}}, status_code=status)


@app.get("/")
def index():
    return FileResponse(BASE / "index.html")


def _http_json(url, headers=None, data=None, timeout=6):
    req = urllib.request.Request(url, headers=headers or {}, data=data)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def search_serper(q, n):
    d = _http_json(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": os.environ["SERPER_API_KEY"], "Content-Type": "application/json"},
        data=json.dumps({"q": q, "num": n, "gl": "es", "hl": "es"}).encode(),
    )
    return [{"title": x.get("title", ""), "url": x.get("link", ""), "snippet": x.get("snippet", "")}
            for x in d.get("organic", [])]


def search_brave(q, n):
    url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode(
        {"q": q, "count": n, "country": "es", "search_lang": "es"})
    d = _http_json(url, headers={"X-Subscription-Token": os.environ["BRAVE_API_KEY"], "Accept": "application/json"})
    return [{"title": x.get("title", ""), "url": x.get("url", ""), "snippet": x.get("description", "")}
            for x in d.get("web", {}).get("results", [])]


def search_tavily(q, n):
    d = _http_json(
        "https://api.tavily.com/search",
        headers={"Authorization": "Bearer " + os.environ["TAVILY_API_KEY"], "Content-Type": "application/json"},
        data=json.dumps({"query": q, "max_results": n, "search_depth": "basic"}).encode(),
    )
    return [{"title": x.get("title", ""), "url": x.get("url", ""), "snippet": x.get("content", "")}
            for x in d.get("results", [])]


def search_searxng(q, n):
    url = os.environ["SEARXNG_URL"].rstrip("/") + "/search?" + urllib.parse.urlencode(
        {"q": q, "format": "json", "language": "es"})
    d = _http_json(url)
    return [{"title": x.get("title", ""), "url": x.get("url", ""), "snippet": x.get("content", "")}
            for x in d.get("results", [])[:n]]


def search_ddg(q, n):
    from ddgs import DDGS
    return [{"title": x.get("title", ""), "url": x.get("href", ""), "snippet": x.get("body", "")}
            for x in DDGS().text(q, region="es-es", max_results=n)]


@app.get("/search")
def search(q: str = "", n: int = 5):
    """Devuelve {results:[{title,url,snippet}]}. Si falla, devuelve lista vacía y el HTML usa Wikipedia."""
    q = q.strip()[:200]
    if not q:
        return {"results": []}
    n = max(1, min(n, 8))
    providers = []
    if os.environ.get("SERPER_API_KEY"):
        providers.append(("serper", search_serper))
    if os.environ.get("TAVILY_API_KEY"):
        providers.append(("tavily", search_tavily))
    if os.environ.get("BRAVE_API_KEY"):
        providers.append(("brave", search_brave))
    if os.environ.get("SEARXNG_URL"):
        providers.append(("searxng", search_searxng))
    providers.append(("duckduckgo", search_ddg))
    for name, fn in providers:
        try:
            res = [r for r in fn(q, n) if r["title"] and r["snippet"]]
            if res:
                return {"results": res, "provider": name}
        except Exception as e:
            print(f"Búsqueda ({name}) falló:", e)
    return {"results": []}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    messages = [
        {"role": m.get("role", "user"), "content": flatten(m.get("content"))}
        for m in body.get("messages", [])
    ]
    if not messages:
        return error("messages vacío")

    temperature = float(body.get("temperature", 0.2))
    top_p = float(body.get("top_p", 1.0))
    rep = float(body.get("repeat_penalty", body.get("repetition_penalty", 1.0)))
    max_new = int(body.get("max_tokens", 320))
    stream = bool(body.get("stream", False))

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    n_prompt = inputs["input_ids"].shape[1]
    if n_prompt + max_new > MAX_CTX:
        # Mismo texto que detecta el HTML para reintentar con menos historial
        return error(f"exceed_context_size: el prompt ({n_prompt}) + respuesta ({max_new}) supera {MAX_CTX} tokens")

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new,
        do_sample=temperature > 0,
        temperature=max(temperature, 0.01),
        top_p=top_p,
        repetition_penalty=rep,
        pad_token_id=tokenizer.eos_token_id,
    )
    cid = "chatcmpl-" + uuid.uuid4().hex[:12]
    created = int(time.time())

    if not stream:
        with LOCK:
            with torch.no_grad():
                out = model.generate(**gen_kwargs)
        text = tokenizer.decode(out[0][n_prompt:], skip_special_tokens=True)
        return {
            "id": cid,
            "object": "chat.completion",
            "created": created,
            "model": MODEL_NAME,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": n_prompt, "completion_tokens": len(out[0]) - n_prompt},
        }

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    def run():
        try:
            with LOCK:
                with torch.no_grad():
                    model.generate(**gen_kwargs, streamer=streamer)
        except Exception as e:  # que el stream no se quede colgado
            print("Error al generar:", e)
            streamer.on_finalized_text("", stream_end=True)

    threading.Thread(target=run, daemon=True).start()

    def chunk(delta, finish=None):
        return "data: " + json.dumps({
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_NAME,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }) + "\n\n"

    def events():
        yield chunk({"role": "assistant"})
        for piece in streamer:
            if piece:
                yield chunk({"content": piece})
        yield chunk({}, "stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)