# Input variables for Legal RAG infrastructure

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "ap-southeast-2"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "legal-rag"
}

# VPC
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones_count" {
  description = "Number of availability zones to use"
  type        = number
  default     = 2
}

# ECS
variable "api_cpu" {
  description = "CPU units for API task (1 vCPU = 1024)"
  type        = number
  default     = 256
}

variable "api_memory" {
  description = "Memory for API task in MB"
  type        = number
  default     = 512
}

variable "qdrant_cpu" {
  description = "CPU units for Qdrant task"
  type        = number
  default     = 512
}

variable "qdrant_memory" {
  description = "Memory for Qdrant task in MB"
  type        = number
  default     = 1024
}

variable "dashboard_cpu" {
  description = "CPU units for Dashboard task"
  type        = number
  default     = 256
}

variable "dashboard_memory" {
  description = "Memory for Dashboard task in MB"
  type        = number
  default     = 512
}

# RDS
variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
}

variable "db_allocated_storage" {
  description = "RDS allocated storage in GB"
  type        = number
  default     = 20
}

variable "db_username" {
  description = "Database master username"
  type        = string
  default     = "legal_rag"
}

variable "db_password" {
  description = "Database master password"
  type        = string
  sensitive   = true
}

# Application
variable "openai_api_key" {
  description = "OpenAI API key"
  type        = string
  sensitive   = true
}

variable "api_image" {
  description = "Docker image for API service"
  type        = string
  default     = ""
}

variable "dashboard_image" {
  description = "Docker image for Dashboard service"
  type        = string
  default     = ""
}
