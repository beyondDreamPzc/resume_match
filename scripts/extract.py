# -*- coding: utf-8 -*-
"""
数据抽取：读取简历纯文本 txt，调用 DeepSeek 文本模型抽取结构化信息，
导出为 resumes.jsonl（含 raw_text 原文 + embed_text 摘要 + 结构化字段）。

用法：
    python scripts/extract.py                 # 默认 --limit 5 小样本
    python scripts/extract.py --limit 0       # 全量
    python scripts/extract.py --limit 5 --txt-dir <dir>

特性：
    - 跳过空文件（<50 字节），记录 skipped 清单。
    - 批次缓存 data/.parse_cache/{file_id}.json，中断续跑不重复调用付费 API。
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

from openai import OpenAI  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(config.BASE_DIR, ".env"))
if not config.DEEPSEEK_API_KEY:
    config.DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")


def natural_key(name: str):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", name)]


def clean_text(text: str) -> str:
    """轻度清洗：去纯符号行、压缩连续空白。"""
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # 去掉纯符号/垃圾行
        if re.fullmatch(r"[~\-—_=|·\s]+", line):
            continue
        lines.append(line)
    return "\n".join(lines)


def extract_one(client: OpenAI, text: str, file_id: str) -> dict:
    """调用 DeepSeek 文本模型抽取结构化信息。"""
    prompt = (
        "你是简历信息抽取助手。请从下面的简历文本中抽取结构化信息，"
        "只返回一个 JSON 对象，不要任何额外文字、不要 markdown 代码块。\n"
        "字段如下（缺失的设为 null）：\n"
        "- name: 姓名\n"
        "- phone: 电话\n"
        "- email: 邮箱\n"
        "- gender: 性别\n"
        "- age: 年龄(数字或 null)\n"
        "- degree: 学历（如 本科/硕士/大专/博士）\n"
        "- school: 毕业院校\n"
        "- major: 专业\n"
        "- work_years: 工作年限（如 \"3年\" 或 \"应届\"）\n"
        "- city: 现居或意向城市\n"
        "- intent: 求职意向岗位（如 运维工程师/Java开发工程师）\n"
        "- summary: 个人简介（1-2 句）\n"
        "- skills: 技能数组，请归一化（如 \"熟恶Java\" 校正为 \"Java\"，去掉错别字）\n"
        "- education: 教育经历数组 [{school, degree, major, period}]\n"
        "- experience: 工作经历数组 [{company, title, period, description}]\n"
        "- projects: 项目经历数组 [{name, role, tech_stack(数组), description}]\n\n"
        f"简历文本：\n{text}"
    )
    resp = client.chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return resp.choices[0].message.content


def parse_json(raw: str) -> dict:
    """多级解析 JSON，失败返回 None。"""
    if not raw:
        return None
    s = raw.strip()
    # 去 markdown 围栏
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        s = s.rsplit("```", 1)[0]
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # 正则提取首个 {...} 块
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def build_embed_text(r: dict) -> str:
    """由结构化字段拼出用于向量化的摘要。"""
    parts = []
    if r.get("intent"):
        parts.append(f"求职意向：{r['intent']}")
    edu = ""
    if r.get("degree") or r.get("school") or r.get("major"):
        edu = "·".join(x for x in [r.get("degree"), r.get("school"), r.get("major")] if x)
    if edu:
        parts.append(f"学历专业：{edu}")
    if r.get("work_years"):
        parts.append(f"工作年限：{r['work_years']}")
    skills = r.get("skills") or []
    if skills:
        parts.append(f"技能：{'、'.join(skills)}")
    for e in (r.get("experience") or []):
        seg = []
        if e.get("title"):
            seg.append(e["title"])
        if e.get("company"):
            seg.append(e["company"])
        desc = e.get("description") or ""
        if seg or desc:
            parts.append(f"工作经历：{'@'.join(seg)}：{desc}")
    for p in (r.get("projects") or []):
        name = p.get("name") or ""
        stack = "、".join(p.get("tech_stack") or [])
        desc = p.get("description") or ""
        parts.append(f"项目：{name}（{stack}）：{desc}")
    if r.get("summary"):
        parts.append(f"简介：{r['summary']}")
    return "\n".join(parts)


def parse_cache_path(file_id: str) -> str:
    os.makedirs(config.PARSE_CACHE_DIR, exist_ok=True)
    return os.path.join(config.PARSE_CACHE_DIR, f"{file_id}.json")


def load_cache(file_id: str):
    p = parse_cache_path(file_id)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_cache(file_id: str, data: dict):
    p = parse_cache_path(file_id)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, p)


def main():
    parser = argparse.ArgumentParser(description="抽取简历结构化信息为 JSONL")
    parser.add_argument("--limit", type=int, default=config.DEFAULT_LIMIT, help="0=全量")
    parser.add_argument("--txt-dir", type=str, default=config.TXT_DIR, help="txt 目录")
    parser.add_argument("--out", type=str, default=config.RESUMES_JSONL, help="输出 jsonl")
    args = parser.parse_args()

    client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)

    txts = sorted([f for f in os.listdir(args.txt_dir) if f.endswith(".txt")], key=natural_key)
    if args.limit > 0:
        txts = txts[: args.limit]

    skipped = []
    rows = []
    for name in txts:
        file_id = name[:-4]  # 去掉 .txt
        path = os.path.join(args.txt_dir, name)
        size = os.path.getsize(path)
        if size < 50:
            skipped.append((name, size))
            print(f"[extract] 跳过空文件 {name} ({size}B)", flush=True)
            continue

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = clean_text(f.read())

        # 缓存命中
        cached = load_cache(file_id)
        if cached is not None:
            rows.append(cached)
            print(f"[extract] {name} 缓存命中", flush=True)
            continue

        # 调用 DeepSeek 抽取
        structured = None
        parse_ok = False
        for attempt in range(3):
            try:
                out = extract_one(client, raw, file_id)
                structured = parse_json(out)
                if structured:
                    parse_ok = True
                    break
            except Exception as e:  # noqa: BLE001
                print(f"[extract] {name} 第{attempt+1}次调用失败: {e}", flush=True)
                time.sleep(1.5 * (attempt + 1))

        if structured is None:
            structured = {}

        row = {
            "id": int(file_id) if file_id.isdigit() else file_id,
            "file_id": name,
            "name": structured.get("name"),
            "phone": structured.get("phone"),
            "email": structured.get("email"),
            "gender": structured.get("gender"),
            "age": structured.get("age"),
            "degree": structured.get("degree"),
            "school": structured.get("school"),
            "major": structured.get("major"),
            "work_years": structured.get("work_years"),
            "city": structured.get("city"),
            "intent": structured.get("intent"),
            "summary": structured.get("summary"),
            "skills": structured.get("skills") or [],
            "education": structured.get("education") or [],
            "experience": structured.get("experience") or [],
            "projects": structured.get("projects") or [],
            "raw_text": raw,
            "parse_ok": parse_ok,
        }
        row["embed_text"] = build_embed_text(row) if parse_ok else raw

        save_cache(file_id, row)
        rows.append(row)
        print(f"[extract] {name} 完成 parse_ok={parse_ok}", flush=True)
        time.sleep(0.4)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[extract] 完成：{len(rows)} 条 → {args.out}")
    if skipped:
        print(f"[extract] 跳过 {len(skipped)} 个空文件: {[s[0] for s in skipped]}")


if __name__ == "__main__":
    main()
