"""
AI 客服 API — FastAPI + pgvector + Gemini

端點：
  POST /chat        — 傳送問題，回傳 AI 回答
  GET  /health      — 健康檢查
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
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

DATABASE_URL  = os.environ["DATABASE_URL"]   # PostgreSQL 連線字串
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"] # Gemini API 金鑰

# ── 模型設定 ──────────────────────────────────────────────────────────────────
EMBED_MODEL = "gemini-embedding-001"  # 將文字轉成向量用的模型（3072 維）
CHAT_MODEL  = "gemini-2.5-flash-lite" # 負責理解問題並生成回答的語言模型
TOP_K = 5  # 向量搜尋時取最相似的前 5 筆資料

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
    """回傳給前端的格式"""
    answer: str        # AI 的回答
    sources: list[str] # 這次回答參考了哪些文件（方便 debug）


# ── 工具函數 ───────────────────────────────────────────────────────────────────

def get_db():
    """建立 PostgreSQL 連線，並啟用 pgvector 支援"""
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)  # 讓 psycopg2 認識 vector 型別
    return conn


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
    """健康檢查，確認服務是否正常運行"""
    return {"status": "ok"}


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
