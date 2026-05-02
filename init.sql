CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id       SERIAL PRIMARY KEY,
    content  TEXT        NOT NULL,
    source   TEXT        NOT NULL,
    embedding vector(3072)
);

