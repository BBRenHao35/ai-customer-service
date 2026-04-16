# AI 客服系統 MVP

基於 RAG（Retrieval-Augmented Generation）架構的 AI 客服系統。
使用者提問 → 從知識庫找相關資料 → 交給 AI 生成回答。

## 畫面截圖

**正常回答：知識庫有的問題**

![正常回答](docs/screenshots/chat-normal.png)

**多輪對話：連續提問，AI 保留對話脈絡**

![多輪對話](docs/screenshots/chat-multi-turn.png)

**知識庫沒有的問題：AI 誠實告知而非亂編**

![無資料回應](docs/screenshots/chat-no-data.png)

## 架構概覽

```
User Question
      │
      ▼
┌─────────────┐   POST /chat   ┌──────────────────────────────┐
│   Frontend  │ ─────────────▶ │    FastAPI (api/main.py)     │
│  (HTML/JS)  │                │                              │
│             │                │  1. embed()                  │
│             │                │     -> Gemini Embedding API  │
│             │                │  2. pgvector search -> Top 5 │
│             │                │  3. build prompt             │
│             │                │     history + context + query│
│             │                │  4. Gemini Chat API          │
│             │                │     -> gemini-2.5-flash-lite │
│             │                │  5. return answer + sources  │
│    Render   │ ◀───────────── │                              │
└─────────────┘                └──────────────┬───────────────┘
                                              │
                                              ▼
                               ┌──────────────────────────────┐
                               │    PostgreSQL + pgvector     │
                               │ content / source / embedding │
                               └──────────────────────────────┘

[準備階段，資料有異動時手動執行]
┌─────────────────────────────────────────────────┐
│ ingest.py: docs/*.txt -> chunk -> embed -> DB   │
└─────────────────────────────────────────────────┘
```

### 架構流程說明

#### 服務階段（使用者提問時）

1. **前端送出請求**：使用者在聊天介面輸入問題，前端以 HTTP POST 傳送問題與歷史對話到 `/chat` API
2. **問題向量化**：後端呼叫 Gemini Embedding API，將問題文字轉換成 3072 維的數字向量，代表這段文字的語意座標
3. **向量相似度搜尋**：拿問題向量去 PostgreSQL（pgvector）比對資料庫中所有文件向量的距離，取出語意最相近的前 5 筆知識庫片段
4. **組裝 Prompt**：將搜尋到的知識庫片段、歷史對話（最近 6 筆）、使用者問題，打包成完整的對話內容傳給 AI
5. **呼叫 Gemini Chat API**：AI 根據提供的知識庫內容生成回答，而不是憑空回答，確保回答來源可追溯
6. **回傳並渲染**：後端將 AI 回答與來源文件名稱回傳給前端，前端渲染成對話訊息顯示給使用者

#### 準備階段（知識庫建立，手動執行一次）

1. **讀取文件**：從 `ingest/docs/` 資料夾讀取所有 `.txt` 知識庫文件
2. **切塊（Chunking）**：將長文件切成較小的片段（每塊約 400 字），相鄰片段保留 50 字重疊避免語意截斷
3. **向量化**：每個片段呼叫 Gemini Embedding API 轉成向量
4. **存入資料庫**：將原始文字、來源檔名、向量一起存入 PostgreSQL，供後續搜尋使用

## 使用工具

| 工具 | 用途 |
|---|---|
| **FastAPI** | Python 後端框架，提供 REST API |
| **PostgreSQL** | 關聯式資料庫，儲存文件內容與向量 |
| **pgvector** | PostgreSQL 擴充套件，支援向量儲存與相似度搜尋 |
| **Gemini API** | Google AI，負責文字向量化與對話生成 |
| **Docker / Docker Compose** | 容器化，一鍵啟動所有服務 |

## 專案結構

```
ai-customer-service/
├── docker-compose.yml       # 定義並啟動所有服務（postgres + api）
├── init.sql                 # 資料庫初始化：建立 table 與 vector index
├── .env.example             # 環境變數範本
│
├── api/                     # 後端服務（Docker 容器）
│   ├── Dockerfile           # 以 python:3.11-slim 為基底建立 image
│   ├── requirements.txt     # Python 套件清單
│   └── main.py              # FastAPI 主程式，RAG 核心邏輯
│
├── ingest/                  # 知識庫建立工具（本機手動執行）
│   ├── requirements.txt     # Python 套件清單
│   ├── ingest.py            # 讀文件 → 切塊 → 向量化 → 存 DB
│   └── docs/                # 放你的知識庫文件（.txt 格式）
│       └── sample_faq.txt   # 範例 FAQ
│
└── frontend/
    └── index.html           # 聊天介面，直接用瀏覽器開啟即可
```

## 前置需求

- Docker & Docker Compose
- Python 3.11+
- Gemini API Key（[Google AI Studio](https://aistudio.google.com) 免費申請）

## 啟動步驟

### 1. Clone 專案

```bash
git clone <your-repo-url>
cd ai-customer-service
```

### 2. 設定環境變數

```bash
cp .env.example .env
```

編輯 `.env`，填入你的 Gemini API Key：

```
GEMINI_API_KEY=AIza你的key
POSTGRES_DB=aics
POSTGRES_USER=aics
POSTGRES_PASSWORD=aics
```

### 3. 啟動資料庫

```bash
docker compose up postgres -d
```

確認啟動成功（Status 顯示 `healthy`）：

```bash
docker compose ps
```

### 4. 準備知識庫文件

把你的 FAQ 或知識文件放入 `ingest/docs/`（`.txt` 格式）。

範例文件 `sample_faq.txt` 已內建，可直接測試。

### 5. 執行 Ingest（建立向量資料庫）

```bash
cd ingest
python3 -m pip install -r requirements.txt

DATABASE_URL=postgresql://aics:aics@localhost:5432/aics python3 ingest.py
```

成功後會看到：
```
舊資料已清除，開始重新 ingest...
[sample_faq.txt] 1 chunks. done
完成！共 ingest 1 個 chunks。
```

> **更新文件時**：修改 `docs/` 內容後重跑此步驟即可，舊資料會自動清除重建。

### 6. 啟動 API

```bash
cd ..  # 回到專案根目錄
docker compose up api -d --build
```

### 7. 開啟前端

```bash
open frontend/index.html
```

或直接在 Finder 點兩下 `index.html`。

---

## 核心流程說明

### RAG 是什麼？

RAG（Retrieval-Augmented Generation）= 先搜尋、再生成。

```
一般 AI：問題 → AI（靠訓練資料回答，可能過時或錯誤）

RAG：   問題 → 搜尋你的知識庫 → 找到相關資料 → AI（根據你的資料回答）
```

AI 本身不知道你公司的 FAQ，但透過 RAG，每次回答前會先查你的資料庫，再根據查到的內容生成回答。

### 向量搜尋是什麼？

文字無法直接比較相似度，但向量可以。

```
"退貨" → [0.023, -0.182, 0.441, ...]  ← 3072 個數字
"商品要怎麼退？" → [0.019, -0.175, 0.438, ...]  ← 這兩個很接近！
"天氣如何" → [0.891, 0.234, -0.102, ...]  ← 這個很遠
```

pgvector 負責計算向量之間的距離，找出語意最相近的文字。

### Ingest vs API 的分工

| | ingest.py | main.py（API）|
|---|---|---|
| 執行時機 | 準備階段，手動跑一次 | 持續運行，每次有人問就執行 |
| 做什麼 | 文件 → 向量 → 存入 DB | 問題 → 搜尋 → AI 回答 |
| 比喻 | 建圖書館（整理書目） | 圖書館員（幫你查書） |

---

## 日後擴充方向

- **新增知識來源**：把 PDF、網頁內容轉成 `.txt` 放入 `docs/`，重跑 `ingest.py`
- **接 LINE Bot**：把 `/chat` API 接上 LINE Messaging API webhook
- **Agent 功能**：讓 AI 可以查訂單、建 ticket 等（需要加 tool calling）
- **提升搜尋品質**：加 reranker 對搜尋結果二次排序
- **對話摘要**：解決長對話歷史截斷問題

---

## 關於這個專案

這是一個學習型專案，用來探索 AI 客服的主流實作方式。

程式碼透過 [Claude Code](https://claude.ai/code) 輔助產生。我自己的背景偏向系統分析與基礎設施，程式撰寫不是強項，但對流程與架構有一定理解。起點是朋友分享的方向與大致流程概念，我從那個起點出發，透過與 AI 討論去理解細節、確認技術選型、釐清各元件的職責——像是 RAG 的完整流程、pgvector 相對於獨立向量資料庫的取捨、MVP 要包含哪些範圍——再從討論結果中判斷哪個方向適合，逐步推進。

這個專案對我來說是一次實作練習，目的是把概念層面的理解落地成一個可以跑起來的系統。
