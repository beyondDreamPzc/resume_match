# -*- coding: utf-8 -*-
"""
匹配核心：embed_query → recall（Milvus 向量召回）→ score_resume（本地 LLM 打分）→ 融合排序。

对外主入口：match(jd, top_n, use_llm) -> dict
"""
import json
import os
import re
import sys
import threading
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

import requests  # noqa: E402
from pymilvus import MilvusClient  # noqa: E402
from openai import OpenAI  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(config.BASE_DIR, ".env"))
if not config.DEEPSEEK_API_KEY:
    config.DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

_client = None
_lock = threading.Lock()
_jd_cache = {}
_deepseek_client = None


def _get_deepseek() -> OpenAI:
    """惰性获取 DeepSeek 云端 client（打分用，输出稳定）。"""
    global _deepseek_client
    if _deepseek_client is None:
        _deepseek_client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
        )
    return _deepseek_client


def get_client() -> MilvusClient:
    global _client
    if _client is None:
        _client = MilvusClient(uri=f"http://{config.MILVUS_HOST}:{config.MILVUS_PORT}")
    return _client


def embed_query(text: str):
    url = f"{config.OLLAMA_BASE_URL}/api/embed"
    resp = requests.post(url, json={"model": config.EMBED_MODEL, "input": [text]}, timeout=120)
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def recall(jd: str, top_n: int) -> list[dict]:
    """向量召回 top_n 份简历，返回含标量字段 + vec_similarity。"""
    client = get_client()
    vec = embed_query(jd)
    results = client.search(
        collection_name=config.COLLECTION_NAME,
        data=[vec],
        limit=top_n,
        output_fields=["id", "file_id", "name", "degree", "school", "major",
                       "work_years", "city", "intent", "skills", "raw_text", "embed_text"],
        search_params=config.SEARCH_PARAM,
    )
    hits = results[0] if results else []
    out = []
    for h in hits:
        e = h.get("entity", {})
        skills = e.get("skills") or ""
        try:
            skills = json.loads(skills) if skills else []
        except (json.JSONDecodeError, TypeError):
            skills = [skills] if skills else []
        out.append({
            "id": e.get("id"),
            "file_id": e.get("file_id", ""),
            "name": e.get("name") or "",
            "degree": e.get("degree") or "",
            "school": e.get("school") or "",
            "major": e.get("major") or "",
            "work_years": e.get("work_years") or "",
            "city": e.get("city") or "",
            "intent": e.get("intent") or "",
            "skills": skills,
            "raw_text": e.get("raw_text") or "",
            "embed_text": e.get("embed_text") or "",
            "vec_similarity": round(h.get("distance", 0.0), 4),
        })
    return out


# ---------------- LLM 打分 ----------------

def _llm_chat(prompt: str) -> str:
    """调用 DeepSeek 云端打分（输出稳定，JSON 可靠）。"""
    client = _get_deepseek()
    resp = client.chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return resp.choices[0].message.content


def parse_score(raw: str, vec_similarity: float) -> dict:
    """多级解析 LLM 输出，返回 {score, reason, matched_skills}。"""
    default = {"score": int(vec_similarity * 100), "reason": "解析失败，使用向量相似度", "matched_skills": []}

    if not raw:
        return default
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        s = s.rsplit("```", 1)[0]
    s = s.strip()

    # 一级：完整 JSON
    try:
        obj = json.loads(s)
        return {
            "score": int(obj.get("score", vec_similarity * 100)),
            "reason": obj.get("reason", "") or "",
            "matched_skills": obj.get("matched_skills", []) or [],
        }
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # 二级：正则提取含 score 的 {...} 块
    m = re.search(r"\{[^{}]*\"score\"[^{}]*\}", s, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            return {
                "score": int(obj.get("score", vec_similarity * 100)),
                "reason": obj.get("reason", "") or "",
                "matched_skills": obj.get("matched_skills", []) or [],
            }
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # 三级：只捞 score 数字
    m2 = re.search(r'"score"\s*:\s*(\d+)', s)
    if m2:
        return {"score": int(m2.group(1)), "reason": s[:80], "matched_skills": []}

    return default


def score_resume(jd: str, resume: dict) -> dict:
    """单份简历打分，返回 {score, reason, matched_skills}（失败降级用向量分）。"""
    prompt = (
        "你是资深 HR / 招聘专家。请评估下面候选人与岗位的匹配程度。\n\n"
        f"【岗位 JD】\n{jd}\n\n"
        f"【候选人简历】\n{resume['raw_text']}\n\n"
        "请严格只输出一行 JSON，不要任何额外文字、不要 markdown 代码块、不要解释：\n"
        '{"score": 0到100的整数, "reason": "不超过80字的匹配理由", "matched_skills": ["匹配的技能1", "技能2"]}\n'
        "评分维度：技能匹配度(40%)、经验相关性(30%)、学历/专业(15%)、岗位意向(15%)。缺失关键技能要明显扣分。"
    )
    try:
        raw = _llm_chat(prompt)
    except Exception as e:  # noqa: BLE001
        print(f"[matcher] 简历 #{resume['id']} 打分失败: {e}", file=sys.stderr)
        return {"score": int(resume["vec_similarity"] * 100),
                "reason": "打分超时，使用向量相似度", "matched_skills": []}
    return parse_score(raw, resume["vec_similarity"])


# ---------------- 匹配入口 ----------------

def match(jd: str, top_n: int = config.TOP_N, use_llm: bool = True) -> dict:
    """完整匹配流程：召回 → 打分 → 融合 → 排序。"""
    jd = (jd or "").strip()
    if not jd:
        return {"error": "请输入岗位 JD", "results": []}

    # 结果缓存（相同 JD 不重复打分）
    cache_key = hashlib.md5(f"{jd}|{top_n}|{use_llm}".encode()).hexdigest()
    if cache_key in _jd_cache:
        return _jd_cache[cache_key]

    candidates = recall(jd, top_n)
    if not candidates:
        return {"error": "简历库为空或未建索引", "results": []}

    if use_llm:
        # 并发打分
        results = []
        with ThreadPoolExecutor(max_workers=config.SCORE_WORKERS) as pool:
            futures = {pool.submit(score_resume, jd, c): c for c in candidates}
            for fut in as_completed(futures):
                c = futures[fut]
                try:
                    scored = fut.result()
                except Exception as e:  # noqa: BLE001
                    scored = {"score": int(c["vec_similarity"] * 100),
                              "reason": "打分异常", "matched_skills": []}
                final = round(config.LLM_WEIGHT * scored["score"]
                              + config.VEC_WEIGHT * c["vec_similarity"] * 100, 1)
                results.append({
                    "id": c["id"],
                    "name": c["name"],
                    "score": final,
                    "llm_score": scored["score"],
                    "vec_similarity": c["vec_similarity"],
                    "reason": scored.get("reason", ""),
                    "matched_skills": scored.get("matched_skills", []),
                    "degree": c["degree"],
                    "work_years": c["work_years"],
                    "city": c["city"],
                    "intent": c["intent"],
                    "skills": c["skills"],
                    "school": c["school"],
                    "major": c["major"],
                    "raw_text": c["raw_text"],
                })
    else:
        # 仅向量排序（快速模式）
        results = [{
            "id": c["id"],
            "name": c["name"],
            "score": round(c["vec_similarity"] * 100, 1),
            "llm_score": None,
            "vec_similarity": c["vec_similarity"],
            "reason": "仅向量相似度排序",
            "matched_skills": [],
            "degree": c["degree"],
            "work_years": c["work_years"],
            "city": c["city"],
            "intent": c["intent"],
            "skills": c["skills"],
            "school": c["school"],
            "major": c["major"],
            "raw_text": c["raw_text"],
        } for c in candidates]

    results.sort(key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(results, 1):
        r["rank"] = i

    payload = {
        "jd": jd,
        "total": get_total_count(),
        "recalled": len(candidates),
        "results": results,
    }

    # 简单缓存淘汰
    if len(_jd_cache) >= config.JD_CACHE_SIZE:
        _jd_cache.clear()
    _jd_cache[cache_key] = payload
    return payload


def get_total_count() -> int:
    """简历库总数。"""
    try:
        stats = get_client().get_collection_stats(config.COLLECTION_NAME)
        return stats.get("row_count", 0)
    except Exception:  # noqa: BLE001
        return 0


def is_ready() -> bool:
    try:
        return get_client().has_collection(config.COLLECTION_NAME)
    except Exception:  # noqa: BLE001
        return False
