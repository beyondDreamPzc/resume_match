# -*- coding: utf-8 -*-
"""
全局配置：简历匹配服务。
Milvus / Ollama / DeepSeek 地址与模型，均支持环境变量覆盖（Docker 容器内需用服务名）。
"""
import os

# ---------- 路径 ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TXT_DIR = os.path.join(BASE_DIR, "data", "txt_output")        # 简历纯文本目录
RESUMES_JSONL = os.path.join(BASE_DIR, "data", "resumes.jsonl")  # 结构化简历
SAMPLE_JDS = os.path.join(BASE_DIR, "data", "sample_jds.json")   # 示例 JD
PARSE_CACHE_DIR = os.path.join(BASE_DIR, "data", ".parse_cache") # DeepSeek 抽取缓存
EMBED_CACHE_DIR = os.path.join(BASE_DIR, "data", ".embed_cache") # 向量化缓存

# ---------- Milvus ----------
MILVUS_HOST = os.environ.get("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = int(os.environ.get("MILVUS_PORT", "19530"))
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "resume")
EMBED_DIM = 1024                      # bge-m3 维度
INDEX_PARAM = {                       # 53 条量小，nlist 调小
    "index_type": "IVF_FLAT",
    "metric_type": "COSINE",
    "params": {"nlist": 16},
}
SEARCH_PARAM = {"metric_type": "COSINE", "params": {"nprobe": 8}}

# ---------- Ollama ----------
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3:latest")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-r1:1.5b")
EMBED_BATCH_SIZE = 16
LLM_TIMEOUT = 60                     # 单份打分超时（秒）

# ---------- DeepSeek（结构化抽取，云端文本模型）----------
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

# ---------- 匹配参数 ----------
TOP_N = 10            # 默认召回数量
LLM_WEIGHT = 0.75     # 融合时 LLM 分权重
VEC_WEIGHT = 0.25     # 融合时向量分权重
SCORE_WORKERS = 3     # 并发打分线程数
JD_CACHE_SIZE = 64    # JD 结果缓存条数

# ---------- 其它 ----------
DEFAULT_LIMIT = 5     # 小样本默认条数；0 表示全量
