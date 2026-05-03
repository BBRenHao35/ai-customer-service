#!/bin/bash
set -e

# 讀取 .env（取得 DATABASE_URL、ADMIN_API_KEY、TELEGRAM_BOT_TOKEN）
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
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
gcloud run deploy "$SERVICE" \
  --image="$IMAGE" \
  --region="$REGION" \
  --platform=managed \
  --allow-unauthenticated \
  --set-env-vars="GEMINI_API_KEY=${GEMINI_API_KEY},DATABASE_URL=${DATABASE_URL},ADMIN_API_KEY=${ADMIN_API_KEY},TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}"

echo "✓ Deploy 完成"
