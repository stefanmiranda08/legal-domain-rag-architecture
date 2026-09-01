# EFS for Qdrant persistent storage

resource "aws_efs_file_system" "qdrant" {
  creation_token = "${var.project_name}-qdrant-efs"
  encrypted      = true

  performance_mode = "generalPurpose"
  throughput_mode  = "bursting"

  lifecycle_policy {
    transition_to_ia = "AFTER_30_DAYS"
  }

  tags = {
    Name = "${var.project_name}-qdrant-efs"
  }
}

resource "aws_efs_mount_target" "qdrant" {
  count = var.availability_zones_count

  file_system_id  = aws_efs_file_system.qdrant.id
  subnet_id       = aws_subnet.private[count.index].id
  security_groups = [aws_security_group.efs.id]
}

resource "aws_efs_access_point" "qdrant" {
  file_system_id = aws_efs_file_system.qdrant.id

  posix_user {
    gid = 1000
    uid = 1000
  }

  root_directory {
    path = "/qdrant"

    creation_info {
      owner_gid   = 1000
      owner_uid   = 1000
      permissions = "755"
    }
  }

  tags = {
    Name = "${var.project_name}-qdrant-ap"
  }
}

# IAM policy for EFS access
resource "aws_iam_role_policy" "ecs_task_efs" {
  name = "${var.project_name}-efs-access"
  role = aws_iam_role.ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "elasticfilesystem:ClientMount",
          "elasticfilesystem:ClientWrite"
        ]
        Resource = aws_efs_file_system.qdrant.arn
        Condition = {
          StringEquals = {
            "elasticfilesystem:AccessPointArn" = aws_efs_access_point.qdrant.arn
          }
        }
      }
    ]
  })
}
