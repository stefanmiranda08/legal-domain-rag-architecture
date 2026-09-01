# Output values

output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}

output "public_subnets" {
  description = "Public subnet IDs"
  value       = aws_subnet.public[*].id
}

output "private_subnets" {
  description = "Private subnet IDs"
  value       = aws_subnet.private[*].id
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.main.name
}

output "ecs_cluster_arn" {
  description = "ECS cluster ARN"
  value       = aws_ecs_cluster.main.arn
}

output "ecr_api_repository_url" {
  description = "ECR repository URL for API"
  value       = aws_ecr_repository.api.repository_url
}

output "ecr_dashboard_repository_url" {
  description = "ECR repository URL for Dashboard"
  value       = aws_ecr_repository.dashboard.repository_url
}

output "alb_dns_name" {
  description = "ALB DNS name"
  value       = aws_lb.main.dns_name
}

output "alb_zone_id" {
  description = "ALB Zone ID (for Route53)"
  value       = aws_lb.main.zone_id
}

output "api_url" {
  description = "API URL"
  value       = "http://${aws_lb.main.dns_name}"
}

output "dashboard_url" {
  description = "Dashboard URL"
  value       = "http://${aws_lb.main.dns_name}/dashboard"
}

output "rds_endpoint" {
  description = "RDS endpoint"
  value       = aws_db_instance.main.endpoint
}

output "rds_database_name" {
  description = "RDS database name"
  value       = aws_db_instance.main.db_name
}

output "efs_file_system_id" {
  description = "EFS file system ID"
  value       = aws_efs_file_system.qdrant.id
}

output "snapshots_bucket" {
  description = "S3 bucket for Qdrant snapshots"
  value       = aws_s3_bucket.snapshots.id
}

output "snapshots_bucket_arn" {
  description = "S3 bucket ARN for Qdrant snapshots"
  value       = aws_s3_bucket.snapshots.arn
}
