"""
ingest.py — 把 docs/ 資料夾內的文件切塊、向量化，存進 PostgreSQL

這個腳本是「準備階段」，只需要在以下情況重新執行：
  - 第一次建立知識庫
  - 新增或修改了 docs/ 裡的文件

用法：
  DATABASE_URL=postgresql://... GEMINI_API_KEY=AIza... python3 ingest.py
  或先建好 .env 再執行：python3 ingest.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from pgvector.psycopg2 import register_vector
from google import genai
from google.genai import types

# ── 環境變數 ──────────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent / ".env")

DATABASE_URL  = os.environ["DATABASE_URL"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

# ── 設定 ──────────────────────────────────────────────────────────────────────
EMBED_MODEL   = "gemini-embedding-001"  # 文字轉向量用的模型（3072 維）
CHUNK_SIZE    = 400   # 每個 chunk 最多幾個 word
CHUNK_OVERLAP = 50    # 相鄰 chunk 重疊幾個 word（避免語意在切割處斷掉）
DOCS_DIR      = Path(__file__).parent / "docs"  # 文件資料夾路徑

# ── 初始化 Gemini 客戶端 ───────────────────────────────────────────────────────
client = genai.Client(api_key=GEMINI_API_KEY)


def chunk_text(text: str) -> list[str]:
    """
    將長文字切成較小的片段（chunks）

    為什麼要切塊？
    - 向量搜尋是「以整段文字為單位」做比對
    - 文字太長，向量會稀釋掉重點，搜尋準確度下降
    - 切成小塊，每塊語意更集中，搜尋更精準

    為什麼要 overlap？
    - 如果一個問題的答案剛好跨在兩個 chunk 的邊界
    - overlap 讓前後兩塊都包含那段文字，不會漏掉

    範例（CHUNK_SIZE=5, OVERLAP=2）：
    原文：[A B C D E F G H]
    chunk1：[A B C D E]
    chunk2：[D E F G H]  ← D E 重複出現
    """
    words = text.split()  # 按空白切成 word list
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        # 下一個 chunk 的起點往回退 CHUNK_OVERLAP 個 word
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def embed(text: str) -> list[float]:
    """
    將一段文字轉換成向量

    task_type="retrieval_document" 代表這是「要被搜尋的文件」
    （相對於查詢時用的 "retrieval_query"）
    Gemini 會針對這個方向最佳化向量的方向性
    """
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type="retrieval_document"),
    )
    return result.embeddings[0].values  # 取出 3072 個浮點數的 list


def main():
    # ── 連線資料庫 ────────────────────────────────────────────────────────────
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)  # 讓 psycopg2 認識 vector 資料型別
    cur = conn.cursor()

    # 清掉舊資料，確保重跑時不會產生重複資料
    cur.execute("DELETE FROM documents")
    conn.commit()
    print("舊資料已清除，開始重新 ingest...\n")

    total_chunks = 0

    # ── 逐一處理 docs/ 裡的每個文件 ──────────────────────────────────────────
    for filepath in DOCS_DIR.iterdir():
        if not filepath.is_file():
            continue

        text = filepath.read_text(encoding="utf-8").strip()
        if not text:
            continue

        # Step 1：把文件切成小塊
        chunks = chunk_text(text)
        print(f"[{filepath.name}] {len(chunks)} chunks", end="", flush=True)

        # Step 2：每個 chunk 轉成向量，存進資料庫
        for chunk in chunks:
            embedding = embed(chunk)  # 呼叫 Gemini API 取得向量
            cur.execute(
                # %s 是安全的參數佔位符，防止 SQL injection
                "INSERT INTO documents (content, source, embedding) VALUES (%s, %s, %s)",
                (chunk, filepath.name, embedding),
            )
            print(".", end="", flush=True)  # 每存一筆印一個點，顯示進度

        conn.commit()  # 每個檔案處理完才 commit，確保資料一致性
        total_chunks += len(chunks)
        print(" done")

    cur.close()
    conn.close()
    print(f"\n完成！共 ingest {total_chunks} 個 chunks。")


if __name__ == "__main__":
    # 只有直接執行這個檔案時才跑 main()
    # 如果是被其他程式 import，則不會自動執行
    main()
