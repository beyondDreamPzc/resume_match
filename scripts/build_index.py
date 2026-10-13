# -*- coding: utf-8 -*-
"""
建索引：读取 resumes.jsonl，用 bge-m3 向量化 embed_text，写入 Milvus collection。

用法（务必 python -u）：
    python -u scripts/build_index.py              # 全量（默认 --limit 0）
    python -u scripts/build_index.py --limit 5    # 小样本
    python -u scripts/build_index.py --drop       # 重建前删旧 collection

特性：批次缓存 data/.embed_cache/NNNNN.json，中断续跑。
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

import requests  # noqa: E402
from pymilvus import (  # noqa: E402
    Collection, CollectionSchema, DataType, FieldSchema,
    connections, utility,
)


# ---------------- Ollama embedding ----------------

def embed_batch(texts):
    url = f"{config.OLLAMA_BASE_URL}/api/embed"
    resp = requests.post(url, json={"model": config.EMBED_MODEL, "input": texts}, timeout=600)
    resp.raise_for_status()
    return [e for e in resp.json()["embeddings"]]


# ---------------- 批次缓存 ----------------

def cache_path(batch_idx):
    os.makedirs(config.EMBED_CACHE_DIR, exist_ok=True)
    return os.path.join(config.EMBED_CACHE_DIR, f"{batch_idx:05d}.json")


def load_cache(batch_idx):
    p = cache_path(batch_idx)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_cache(batch_idx, vectors):
    p = cache_path(batch_idx)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(vectors, f)
    os.replace(tmp, p)


# ---------------- Milvus ----------------

def ensure_collection(drop: bool = False) -> Collection:
    connections.connect(alias="default", host=config.MILVUS_HOST, port=config.MILVUS_PORT)

    if drop and utility.has_collection(config.COLLECTION_NAME):
        utility.drop_collection(config.COLLECTION_NAME)
        print(f"[index] 已删除旧 collection '{config.COLLECTION_NAME}'")

    if utility.has_collection(config.COLLECTION_NAME):
        print(f"[index] collection '{config.COLLECTION_NAME}' 已存在，直接复用")
        return Collection(config.COLLECTION_NAME)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="file_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="degree", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="school", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="major", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="work_years", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="city", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="intent", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="skills", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="raw_text", dtype=DataType.VARCHAR, max_length=60000),
        FieldSchema(name="embed_text", dtype=DataType.VARCHAR, max_length=30000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=config.EMBED_DIM),
    ]
    schema = CollectionSchema(fields, description="简历库")
    col = Collection(config.COLLECTION_NAME, schema=schema)
    col.create_index(field_name="embedding", index_params=config.INDEX_PARAM)
    print(f"[index] 已创建 collection '{config.COLLECTION_NAME}' 并建立索引")
    return col


def s(v):
    """标量字段安全转字符串（None→''，list→JSON字符串）。"""
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def main():
    parser = argparse.ArgumentParser(description="向量化 resumes.jsonl 并写入 Milvus")
    parser.add_argument("--jsonl", type=str, default=config.RESUMES_JSONL, help="输入 jsonl")
    parser.add_argument("--limit", type=int, default=0, help="只建前 N 条，0=全部")
    parser.add_argument("--drop", action="store_true", help="重建前删旧 collection")
    args = parser.parse_args()

    with open(args.jsonl, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        rows = rows[: args.limit]

    print(f"[index] 待向量化 {len(rows)} 条")

    col = ensure_collection(drop=args.drop)
    col.load()

    total = len(rows)
    batch = config.EMBED_BATCH_SIZE
    inserted = 0

    for start in range(0, total, batch):
        end = min(start + batch, total)
        batch_idx = start // batch
        chunk = rows[start:end]

        vectors = load_cache(batch_idx)
        if vectors is None:
            # 向量化 embed_text，截断到 6000 字符避免过长稀释语义
            texts = [r["embed_text"][:6000] for r in chunk]
            vectors = embed_batch(texts)
            save_cache(batch_idx, vectors)

        entities = [
            [int(r["id"]) for r in chunk],
            [r["file_id"] for r in chunk],
            [s(r.get("name")) for r in chunk],
            [s(r.get("degree")) for r in chunk],
            [s(r.get("school")) for r in chunk],
            [s(r.get("major")) for r in chunk],
            [s(r.get("work_years")) for r in chunk],
            [s(r.get("city")) for r in chunk],
            [s(r.get("intent")) for r in chunk],
            [s(r.get("skills")) for r in chunk],
            [s(r.get("raw_text")) for r in chunk],
            [s(r.get("embed_text")) for r in chunk],
            vectors,
        ]
        col.insert(entities)
        inserted += len(chunk)
        print(f"[index] 进度 {inserted}/{total}", flush=True)

    col.flush()
    print(f"[index] 完成：共写入 {inserted} 条 → collection '{config.COLLECTION_NAME}'")


if __name__ == "__main__":
    main()
