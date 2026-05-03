#!/bin/bash
set -e

# 讀取 .env（取得 DATABASE_URL、ADMIN_API_KEY、TELEGRAM_BOT_TOKEN）
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

# 設定
PROJECT_ID="renhao-dev"
REGION="asia-east1"
IMAGE="asia-east1-docker.pkg.dev/${PROJECT_ID}/ai-customer-service/api:latest"
SERVICE="ai-customer-service"

echo "▶ Building image..."
docker build --platform linux/amd64 -t "$IMAGE" ./api

echo "▶ Pushing image..."
docker push "$IMAGE"

echo "▶ Deploying to Cloud Run..."

# 把環境變數寫進暫存 YAML，避免 shell 特殊字元解析問題
ENV_FILE=$(mktemp /tmp/cloudrun-env-XXXXXX.yaml)
cat > "$ENV_FILE" <<EOF
GEMINI_API_KEY: "${GEMINI_API_KEY}"
DATABASE_URL: "${DATABASE_URL}"
ADMIN_API_KEY: "${ADMIN_API_KEY}"
TELEGRAM_BOT_TOKEN: "${TELEGRAM_BOT_TOKEN}"
EOF

gcloud run deploy "$SERVICE" \
  --image="$IMAGE" \
  --region="$REGION" \
  --platform=managed \
  --allow-unauthenticated \
  --env-vars-file="$ENV_FILE"

rm -f "$ENV_FILE"

echo "✓ Deploy 完成"
