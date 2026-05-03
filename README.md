# AI 客服系統

基於 RAG（Retrieval-Augmented Generation）架構的 AI 客服系統。
使用者提問 → 從知識庫找相關資料 → 交給 AI 生成回答。

**Live Demo：** https://BBRenHao35.github.io/ai-customer-service/

## 畫面截圖

**正常回答：知識庫有的問題**

![正常回答](docs/screenshots/chat-normal.png)

**多輪對話：連續提問，AI 保留對話脈絡**

![多輪對話](docs/screenshots/chat-multi-turn.png)

**知識庫沒有的問題：AI 誠實告知而非亂編**

![無資料回應](docs/screenshots/chat-no-data.png)

## 架構

```
使用者瀏覽器
      │
      ▼
┌─────────────────────┐   POST /chat    ┌──────────────────────────────┐
│     GitHub Pages    │ ──────────────▶ │   GCP Cloud Run (FastAPI)    │
│   (靜態前端介面)     │                │                              │
│                     │                │  1. embed()                  │
│                     │                │     -> Gemini Embedding API  │
│                     │                │  2. pgvector search -> Top 5 │
│                     │                │  3. build prompt             │
│                     │                │  4. Gemini Chat API          │
│                     │                │     -> gemini-2.5-flash-lite │
│       Render        │ ◀───────────── │  5. return answer + sources  │
└─────────────────────┘                └──────────────┬───────────────┘
                                                      │
                                                      ▼
                                       ┌──────────────────────────────┐
                                       │   Supabase (PostgreSQL)      │
                                       │   + pgvector extension       │
                                       │   content / source /         │
                                       │   embedding (3072 維)        │
                                       └──────────────────────────────┘

[知識庫管理 API，需 X-Admin-Key]
POST   /admin/ingest              → 上傳文件，自動切塊向量化存入 DB
GET    /admin/documents           → 列出目前所有知識庫內容
DELETE /admin/documents/{id}      → 刪除單一 chunk
DELETE /admin/sources/{source}    → 刪除整份文件的所有 chunks
```

## 使用工具

| 工具 | 用途 |
|---|---|
| **FastAPI** | Python 後端框架，提供 REST API |
| **GCP Cloud Run** | 無伺服器容器平台，部署 FastAPI |
| **Supabase** | 雲端 PostgreSQL，儲存文件內容與向量 |
| **pgvector** | PostgreSQL 擴充套件，支援向量儲存與相似度搜尋 |
| **Gemini API** | Google AI，負責文字向量化（embedding）與對話生成 |
| **GitHub Pages** | 靜態前端託管 |
| **Docker** | 容器化，建立 Cloud Run 部署用的 image |

## 專案結構

```
ai-customer-service/
├── docker-compose.yml       # 本地開發用（postgres + api）
├── init.sql                 # 資料庫初始化 SQL
├── .env.example             # 環境變數範本
│
├── api/                     # 後端服務
│   ├── Dockerfile           # Cloud Run 部署用的 image
│   ├── requirements.txt     # Python 套件
│   └── main.py              # FastAPI 主程式：RAG + 管理 API
│
├── ingest/                  # 本地知識庫建立工具
│   ├── ingest.py            # 讀文件 → 切塊 → 向量化 → 存 DB
│   └── docs/                # 知識庫文件（.txt 格式）
│       └── sample_faq.txt
│
└── docs/                    # GitHub Pages 前端（同時存放截圖）
    ├── index.html           # 聊天介面
    └── screenshots/
```

## 環境變數

| 變數 | 說明 |
|---|---|
| `GEMINI_API_KEY` | Google AI Studio 取得 |
| `DATABASE_URL` | Supabase 的 PostgreSQL 連線字串 |
| `ADMIN_API_KEY` | 管理 API 的保護金鑰（自訂） |

## 部署架構說明

### 資料庫：Supabase

1. 在 [supabase.com](https://supabase.com) 建立 project
2. 在 SQL Editor 執行 `init.sql`
3. 取得 Session Pooler 的連線字串作為 `DATABASE_URL`

### 後端：GCP Cloud Run

```bash
# Build image（指定 linux/amd64，Cloud Run 需要）
docker build --platform linux/amd64 \
  -t asia-east1-docker.pkg.dev/PROJECT_ID/ai-customer-service/api:latest ./api

# Push 到 Artifact Registry
docker push asia-east1-docker.pkg.dev/PROJECT_ID/ai-customer-service/api:latest

# Deploy
gcloud run deploy ai-customer-service \
  --image=asia-east1-docker.pkg.dev/PROJECT_ID/ai-customer-service/api:latest \
  --region=asia-east1 \
  --platform=managed \
  --allow-unauthenticated \
  --set-env-vars="GEMINI_API_KEY=...,DATABASE_URL=...,ADMIN_API_KEY=..."
```

### 前端：GitHub Pages

repo → Settings → Pages → Branch: `main`, Folder: `/docs`

### 知識庫初始化

```bash
cd ingest
pip install -r requirements.txt
DATABASE_URL="your-supabase-url" python3 ingest.py
```

## 管理 API 使用方式

所有管理端點需在 Header 帶 `X-Admin-Key`。

**上傳新文件：**
```bash
curl -X POST https://your-cloud-run-url/admin/ingest \
  -H "X-Admin-Key: your-admin-key" \
  -H "Content-Type: application/json" \
  -d '{"source": "faq.txt", "content": "你的文件內容..."}'
```

**查看知識庫內容：**
```bash
curl https://your-cloud-run-url/admin/documents \
  -H "X-Admin-Key: your-admin-key"
```

**刪除整份文件：**
```bash
curl -X DELETE https://your-cloud-run-url/admin/sources/faq.txt \
  -H "X-Admin-Key: your-admin-key"
```

## 核心概念

### RAG 是什麼？

```
一般 AI：問題 → AI（靠訓練資料回答，可能過時或錯誤）
RAG：    問題 → 搜尋知識庫 → 找到相關資料 → AI（根據你的資料回答）
```

### 向量搜尋是什麼？

文字無法直接比較相似度，但向量可以：

```
"退貨"        → [0.023, -0.182, ...]  ← 3072 個數字
"商品怎麼退？" → [0.019, -0.175, ...]  ← 這兩個很接近 → 搜尋命中
"天氣如何"    → [0.891,  0.234, ...]  ← 這個很遠 → 不會被找到
```

## 日後擴充方向

- **多租戶**：不同客戶有獨立的知識庫（加 `tenant_id`）
- **Observability**：記錄每次對話的 response time、RAG 命中率
- **LINE Bot 整合**：把 `/chat` 接上 LINE Messaging API webhook
- **PDF 支援**：ingest 支援直接上傳 PDF
- **Reranker**：對向量搜尋結果做二次排序，提升回答品質

---

## 關於這個專案

這是一個學習型專案，用來探索 AI 客服的主流實作方式。

程式碼透過 [Claude Code](https://claude.ai/code) 輔助產生。我自己的背景偏向系統分析與基礎設施，程式撰寫不是強項，但對流程與架構有一定理解。起點是朋友分享的方向與大致流程概念，我從那個起點出發，透過與 AI 討論去理解細節、確認技術選型、釐清各元件的職責，再從討論結果中判斷哪個方向適合，逐步推進。
