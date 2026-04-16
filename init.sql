CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id       SERIAL PRIMARY KEY,
    content  TEXT        NOT NULL,
    source   TEXT        NOT NULL,
    embedding vector(3072)
);

-- 加速向量搜尋的 index（資料量大時很重要）
CREATE INDEX IF NOT EXISTS documents_embedding_idx
    ON documents USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
