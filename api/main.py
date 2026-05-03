"""
AI 客服 API — FastAPI + pgvector + Gemini

端點：
  POST /chat        — 傳送問題，回傳 AI 回答
  GET  /health      — 健康檢查
"""

import os
import httpx
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2
from pgvector.psycopg2 import register_vector
from google import genai
from google.genai import types

# ── 環境變數 ──────────────────────────────────────────────────────────────────
# 從專案根目錄的 .env 讀取設定（本機開發用）
# Docker 環境下這些值會由 docker-compose.yml 的 environment 傳入
load_dotenv(Path(__file__).parent.parent / ".env")

DATABASE_URL        = os.environ["DATABASE_URL"]
GEMINI_API_KEY      = os.environ["GEMINI_API_KEY"]
ADMIN_API_KEY       = os.environ["ADMIN_API_KEY"]
TELEGRAM_BOT_TOKEN  = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_API        = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# ── 模型設定 ──────────────────────────────────────────────────────────────────
EMBED_MODEL  = "gemini-embedding-001"
CHAT_MODEL   = "gemini-2.5-flash-lite"
TOP_K        = 5
CHUNK_SIZE   = 400
CHUNK_OVERLAP = 50

# ── 初始化 Gemini 客戶端 ───────────────────────────────────────────────────────
client = genai.Client(api_key=GEMINI_API_KEY)

# ── System Prompt ─────────────────────────────────────────────────────────────
# 告訴 AI 它的角色與行為規則
# 這段文字每次呼叫 AI 都會帶著，等於是給 AI 的「工作說明書」
SYSTEM_PROMPT = """你是一個專業的客服助理。
請根據以下提供的知識庫內容回答用戶的問題。

規則：
1. 只根據知識庫內容回答，不要編造資訊
2. 如果知識庫沒有相關資訊，誠實告知用戶並建議聯繫人工客服
3. 回答要簡潔清楚，使用繁體中文
4. 如果問題模糊，可以請用戶說明得更清楚
"""

# ── FastAPI 初始化 ─────────────────────────────────────────────────────────────
app = FastAPI(title="AI Customer Service")

# CORS 設定：允許前端（瀏覽器直接開 HTML 檔）跨來源呼叫 API
# 正式環境應把 allow_origins 改成指定網域，不要用 "*"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 資料結構定義（Pydantic）────────────────────────────────────────────────────
# Pydantic 會自動驗證前端傳來的 JSON 格式是否正確

class Message(BaseModel):
    """單一對話訊息"""
    role: str     # "user"（使用者）或 "assistant"（AI 回覆）
    content: str  # 訊息內容


class ChatRequest(BaseModel):
    """前端送來的請求格式"""
    message: str              # 這次的問題
    history: list[Message] = []  # 先前的對話紀錄（前端負責保存並每次帶來）


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


# ── 管理 API 的資料結構 ────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    source: str   # 文件名稱，例如 "faq.txt"
    content: str  # 文件內容

class IngestResponse(BaseModel):
    source: str
    chunks_inserted: int

class DocumentItem(BaseModel):
    id: int
    source: str
    content_preview: str  # 只回傳前 100 字，避免回應太大

class DocumentsResponse(BaseModel):
    documents: list[DocumentItem]
    total: int


# ── 工具函數 ───────────────────────────────────────────────────────────────────

def get_db():
    """建立 PostgreSQL 連線，並啟用 pgvector 支援"""
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)  # 讓 psycopg2 認識 vector 型別
    return conn


def verify_admin(x_admin_key: str = Header(...)):
    """管理端點的 API Key 驗證"""
    if x_admin_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin key")


def chunk_text(text: str) -> list[str]:
    """將長文字切成有重疊的小塊，提升向量搜尋準確度"""
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def embed_document(text: str) -> list[float]:
    """將文件文字轉成向量（ingest 用）"""
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type="retrieval_document"),
    )
    return result.embeddings[0].values or []


def embed(text: str) -> list[float]:
    """
    將一段文字轉換成向量（embedding）

    輸入：一段文字字串
    輸出：3072 個浮點數組成的 list，代表這段文字的語意座標

    task_type="retrieval_query" 代表這是「搜尋用的問題」
    （相對於 ingest 時用的 "retrieval_document"）
    Gemini 根據這個提示做不同方向的最佳化
    """
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type="retrieval_query"),
    )
    embeddings = result.embeddings
    assert embeddings is not None and len(embeddings) > 0
    return embeddings[0].values or []


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    向量搜尋：找出知識庫中與問題最相似的文字片段

    流程：
    1. 把問題轉成向量
    2. 用餘弦距離（cosine distance）比對資料庫裡所有向量
    3. 回傳距離最小（最相似）的前 top_k 筆
    """
    # 先把問題也轉成向量，才能跟資料庫的向量做比較
    query_embedding = embed(query)

    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT content, source, 1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        ORDER BY embedding <=> %s::vector  -- 距離小的排前面（最相似）
        LIMIT %s                           -- 只取前 N 筆
        """,
        # %s 是安全的參數佔位符，防止 SQL injection
        (query_embedding, query_embedding, top_k),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    # 整理成 dict list，方便後續使用
    # similarity：1 = 完全相同，0 = 完全不相關
    return [{"content": row[0], "source": row[1], "similarity": row[2]} for row in rows]


def build_contents(history: list[Message], context: str, question: str):
    """
    組裝傳給 Gemini 的對話內容

    Gemini 的格式與 OpenAI 不同：
    - role 用 "user" / "model"（OpenAI 用 "user" / "assistant"）
    - 內容放在 parts 陣列裡

    最終組出來的結構：
    [
        {"role": "user",  "parts": [{"text": "上一輪的問題"}]},
        {"role": "model", "parts": [{"text": "上一輪的回答"}]},
        ...
        {"role": "user",  "parts": [{"text": "知識庫內容 + 這次的問題"}]},
    ]
    """
    contents = []

    # 加入歷史對話，只取最近 6 筆（3 來 3 往）
    # 避免對話太長造成 token 費用過高或超過模型上限
    for msg in history[-6:]:
        # 前端用 "assistant"，Gemini 要用 "model"，這裡做轉換
        role = "model" if msg.role == "assistant" else "user"
        contents.append(types.Content(
            role=role,
            parts=[types.Part(text=msg.content)]
        ))

    # 把從資料庫找到的相關內容（context）跟問題一起組成最後的 user 訊息
    # AI 看到這個才知道要根據哪些資料來回答
    user_content = f"""知識庫內容：
---
{context}
---

用戶問題：{question}"""
    contents.append(types.Content(
        role="user",
        parts=[types.Part(text=user_content)]
    ))
    return contents


# ── API 端點 ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ── 管理 API（需要 X-Admin-Key header）────────────────────────────────────────

@app.post("/admin/ingest", response_model=IngestResponse)
def admin_ingest(req: IngestRequest, x_admin_key: str = Header(...)):
    """
    上傳文件內容到知識庫。
    同名來源會先刪除再重建，方便更新單一文件而不影響其他資料。
    """
    verify_admin(x_admin_key)

    chunks = chunk_text(req.content)
    if not chunks:
        raise HTTPException(status_code=400, detail="content 不可為空")

    conn = get_db()
    cur = conn.cursor()

    # 刪除同名來源的舊資料，讓更新同一份文件不會產生重複
    cur.execute("DELETE FROM documents WHERE source = %s", (req.source,))

    for chunk in chunks:
        embedding = embed_document(chunk)
        cur.execute(
            "INSERT INTO documents (content, source, embedding) VALUES (%s, %s, %s)",
            (chunk, req.source, embedding),
        )

    conn.commit()
    cur.close()
    conn.close()

    return IngestResponse(source=req.source, chunks_inserted=len(chunks))


@app.get("/admin/documents", response_model=DocumentsResponse)
def admin_list_documents(x_admin_key: str = Header(...)):
    """列出目前知識庫的所有文件"""
    verify_admin(x_admin_key)

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, source, content FROM documents ORDER BY source, id")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    docs = [
        DocumentItem(id=r[0], source=r[1], content_preview=r[2][:100])
        for r in rows
    ]
    return DocumentsResponse(documents=docs, total=len(docs))


@app.delete("/admin/documents/{doc_id}")
def admin_delete_document(doc_id: int, x_admin_key: str = Header(...)):
    """刪除單一文件 chunk"""
    verify_admin(x_admin_key)

    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM documents WHERE id = %s RETURNING id", (doc_id,))
    deleted = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()

    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted_id": doc_id}


@app.delete("/admin/sources/{source}")
def admin_delete_source(source: str, x_admin_key: str = Header(...)):
    """刪除某個來源的所有 chunks（例如整份 faq.txt）"""
    verify_admin(x_admin_key)

    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM documents WHERE source = %s RETURNING id", (source,))
    deleted = cur.fetchall()
    conn.commit()
    cur.close()
    conn.close()

    return {"source": source, "deleted_count": len(deleted)}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    主要對話端點，完整 RAG 流程：
    1. 向量搜尋（從知識庫找相關資料）
    2. 組裝 prompt（把資料+問題+歷史對話打包）
    3. 呼叫 Gemini（讓 AI 根據資料生成回答）
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message 不可為空")

    # Step 1：向量搜尋，找出最相關的知識庫內容
    docs = retrieve(req.message)
    if not docs:
        return ChatResponse(
            answer="抱歉，目前知識庫還沒有相關資料，請聯繫人工客服協助您。",
            sources=[],
        )

    # Step 2：把搜尋到的多筆資料合併成一段 context 文字
    context = "\n\n".join(
        f"[來源: {d['source']}]\n{d['content']}" for d in docs
    )
    # 去除重複的來源檔案名稱（set 去重）
    sources = list({d["source"] for d in docs})

    # Step 3：組裝對話內容，呼叫 Gemini 生成回答
    contents = build_contents(req.history, context, req.message)
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,  # 0 = 最穩定保守，1 = 最有創意，客服取低值
        ),
    )
    answer = response.text or ""

    return ChatResponse(answer=answer, sources=sources)


# ── Telegram Bot ───────────────────────────────────────────────────────────────

def send_telegram_message(chat_id: int, text: str):
    """呼叫 Telegram API 把訊息傳給使用者"""
    with httpx.Client() as client_http:
        client_http.post(
            f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": chat_id, "text": text},
        )


@app.post("/telegram/webhook")
def telegram_webhook(payload: dict):
    """
    Telegram 把使用者訊息轉發到這個端點。
    取出問題 → 走 RAG 流程 → 回傳答案給使用者。
    """
    message = payload.get("message", {})
    chat_id = message.get("chat", {}).get("id")
    text = message.get("text", "").strip()

    # 忽略沒有文字的訊息（例如貼圖、圖片）
    if not chat_id or not text:
        return {"ok": True}

    # 走既有的 RAG 流程
    docs = retrieve(text)
    if not docs:
        send_telegram_message(chat_id, "抱歉，目前知識庫還沒有相關資料，請聯繫人工客服協助您。")
        return {"ok": True}

    context = "\n\n".join(f"[來源: {d['source']}]\n{d['content']}" for d in docs)
    contents = build_contents([], context, text)
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,
        ),
    )
    answer = response.text or "抱歉，發生錯誤，請稍後再試。"
    send_telegram_message(chat_id, answer)
    return {"ok": True}
