variable "project_id" {
  default = "renhao-dev"
}

variable "region" {
  default = "asia-east1"
}

variable "gemini_api_key" {
  sensitive = true
}

variable "database_url" {
  sensitive = true
}

variable "admin_api_key" {
  sensitive = true
}

variable "telegram_bot_token" {
  sensitive = true
}
