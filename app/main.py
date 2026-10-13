# -*- coding: utf-8 -*-
"""
FastAPI 后端：简历匹配筛选服务。

启动：
    uvicorn app.main:app --host 127.0.0.1 --port 8000
接口：
    GET  /              -> 前端页面
    GET  /api/health    -> 健康检查
    POST /api/match     -> {"jd": "...", "top_n": 10, "use_llm": true}
    GET  /api/examples  -> 内置示例 JD
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from app.matcher import match, is_ready, get_total_count  # noqa: E402


app = FastAPI(title="简历匹配筛选服务", version="1.0.0")

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


class MatchRequest(BaseModel):
    jd: str
    top_n: int = config.TOP_N
    use_llm: bool = True


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "collection_ready": is_ready(),
        "collection": config.COLLECTION_NAME,
        "resume_count": get_total_count(),
    }


@app.get("/api/examples")
def examples():
    if os.path.exists(config.SAMPLE_JDS):
        with open(config.SAMPLE_JDS, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


@app.post("/api/match")
def api_match(req: MatchRequest):
    t0 = time.time()
    result = match(req.jd, top_n=req.top_n, use_llm=req.use_llm)
    result["elapsed"] = round(time.time() - t0, 2)
    return result


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/favicon.ico")
def favicon():
    from fastapi import Response
    return Response(status_code=204)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
