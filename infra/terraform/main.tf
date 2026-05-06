terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# ── Artifact Registry ──────────────────────────────────────────────────────────
# 存放 Docker image 的倉庫
resource "google_artifact_registry_repository" "api" {
  repository_id = "ai-customer-service"
  format        = "DOCKER"
  location      = var.region
  description   = "AI Customer Service"
}

# ── Cloud Run Service ──────────────────────────────────────────────────────────
# 跑 FastAPI 的無伺服器容器
resource "google_cloud_run_v2_service" "api" {
  name     = "ai-customer-service"
  location = var.region

  template {
    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/ai-customer-service/api:latest"

      env {
        name  = "GEMINI_API_KEY"
        value = var.gemini_api_key
      }
      env {
        name  = "DATABASE_URL"
        value = var.database_url
      }
      env {
        name  = "ADMIN_API_KEY"
        value = var.admin_api_key
      }
      env {
        name  = "TELEGRAM_BOT_TOKEN"
        value = var.telegram_bot_token
      }
    }
  }
}

# 讓所有人都能呼叫 Cloud Run（不需要登入）
resource "google_cloud_run_v2_service_iam_member" "public_access" {
  name     = google_cloud_run_v2_service.api.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ── GitHub Actions Service Account ────────────────────────────────────────────
# 讓 GitHub Actions 有權限 deploy 的 service account
resource "google_service_account" "github_actions" {
  account_id   = "github-actions"
  display_name = "GitHub Actions Deploy"
}

resource "google_project_iam_member" "github_actions_run_admin" {
  project = var.project_id
  role    = "roles/run.admin"
  member  = "serviceAccount:${google_service_account.github_actions.email}"
}

resource "google_project_iam_member" "github_actions_ar_writer" {
  project = var.project_id
  role    = "roles/artifactregistry.writer"
  member  = "serviceAccount:${google_service_account.github_actions.email}"
}

resource "google_project_iam_member" "github_actions_sa_user" {
  project = var.project_id
  role    = "roles/iam.serviceAccountUser"
  member  = "serviceAccount:${google_service_account.github_actions.email}"
}
