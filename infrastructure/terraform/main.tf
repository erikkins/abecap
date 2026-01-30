# Stocker AWS Infrastructure - Terraform
# 
# This creates:
# - S3 bucket for frontend (static website)
# - CloudFront distribution (CDN)
# - Lambda function (backend API)
# - API Gateway (REST API)
# - RDS PostgreSQL (database)
# - EventBridge (scheduled scans)

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  required_version = ">= 1.0"
}

provider "aws" {
  region = var.aws_region
}

# ============================================================================
# Variables
# ============================================================================

variable "aws_region" {
  default = "us-east-1"
}

variable "project_name" {
  default = "stocker"
}

variable "environment" {
  default = "prod"
}

variable "db_password" {
  description = "RDS database password"
  sensitive   = true
}

locals {
  prefix = "${var.project_name}-${var.environment}"
}

# ============================================================================
# S3 Bucket - Frontend Static Site
# ============================================================================

resource "aws_s3_bucket" "frontend" {
  bucket = "${local.prefix}-frontend"
}

resource "aws_s3_bucket_website_configuration" "frontend" {
  bucket = aws_s3_bucket.frontend.id

  index_document {
    suffix = "index.html"
  }

  error_document {
    key = "index.html"
  }
}

resource "aws_s3_bucket_public_access_block" "frontend" {
  bucket = aws_s3_bucket.frontend.id

  block_public_acls       = false
  block_public_policy     = false
  ignore_public_acls      = false
  restrict_public_buckets = false
}

resource "aws_s3_bucket_policy" "frontend" {
  bucket = aws_s3_bucket.frontend.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "PublicReadGetObject"
        Effect    = "Allow"
        Principal = "*"
        Action    = "s3:GetObject"
        Resource  = "${aws_s3_bucket.frontend.arn}/*"
      }
    ]
  })

  depends_on = [aws_s3_bucket_public_access_block.frontend]
}

# ============================================================================
# CloudFront Distribution
# ============================================================================

resource "aws_cloudfront_distribution" "frontend" {
  enabled             = true
  default_root_object = "index.html"
  price_class         = "PriceClass_100"

  origin {
    domain_name = aws_s3_bucket_website_configuration.frontend.website_endpoint
    origin_id   = "S3-${aws_s3_bucket.frontend.id}"

    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "http-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  default_cache_behavior {
    allowed_methods        = ["GET", "HEAD"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "S3-${aws_s3_bucket.frontend.id}"
    viewer_protocol_policy = "redirect-to-https"

    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }

    min_ttl     = 0
    default_ttl = 3600
    max_ttl     = 86400
  }

  # SPA routing - return index.html for 404s
  custom_error_response {
    error_code         = 404
    response_code      = 200
    response_page_path = "/index.html"
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }

  tags = {
    Name = "${local.prefix}-frontend"
  }
}

# ============================================================================
# Lambda Function - Backend API
# ============================================================================

# IAM Role for Lambda
resource "aws_iam_role" "lambda" {
  name = "${local.prefix}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy_attachment" "lambda_vpc" {
  role       = aws_iam_role.lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole"
}

# Lambda Function
resource "aws_lambda_function" "api" {
  filename         = "lambda.zip"  # Created by CI/CD
  function_name    = "${local.prefix}-api"
  role             = aws_iam_role.lambda.arn
  handler          = "main.handler"
  runtime          = "python3.11"
  timeout          = 30
  memory_size      = 512

  environment {
    variables = {
      DATABASE_URL = "postgresql://${aws_db_instance.main.username}:${var.db_password}@${aws_db_instance.main.endpoint}/${aws_db_instance.main.db_name}"
      ENVIRONMENT  = var.environment
    }
  }

  tags = {
    Name = "${local.prefix}-api"
  }
}

# ============================================================================
# API Gateway
# ============================================================================

resource "aws_apigatewayv2_api" "main" {
  name          = "${local.prefix}-api"
  protocol_type = "HTTP"

  cors_configuration {
    allow_origins = ["*"]
    allow_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    allow_headers = ["*"]
  }
}

resource "aws_apigatewayv2_integration" "lambda" {
  api_id             = aws_apigatewayv2_api.main.id
  integration_type   = "AWS_PROXY"
  integration_uri    = aws_lambda_function.api.invoke_arn
  integration_method = "POST"
}

resource "aws_apigatewayv2_route" "default" {
  api_id    = aws_apigatewayv2_api.main.id
  route_key = "$default"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.main.id
  name        = "$default"
  auto_deploy = true
}

resource "aws_lambda_permission" "api_gateway" {
  statement_id  = "AllowAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.main.execution_arn}/*/*"
}

# ============================================================================
# RDS PostgreSQL
# ============================================================================

resource "aws_db_instance" "main" {
  identifier           = "${local.prefix}-db"
  engine               = "postgres"
  engine_version       = "15"
  instance_class       = "db.t3.micro"
  allocated_storage    = 20
  storage_type         = "gp2"
  db_name              = "stocker"
  username             = "stocker"
  password             = var.db_password
  publicly_accessible  = false
  skip_final_snapshot  = true

  tags = {
    Name = "${local.prefix}-db"
  }
}

# ============================================================================
# EventBridge - Scheduled Scanner
# ============================================================================

resource "aws_cloudwatch_event_rule" "scanner" {
  name                = "${local.prefix}-scanner"
  description         = "Run market scan at 4 PM ET on weekdays"
  schedule_expression = "cron(0 21 ? * MON-FRI *)"  # 4 PM ET = 9 PM UTC
}

resource "aws_cloudwatch_event_target" "scanner" {
  rule      = aws_cloudwatch_event_rule.scanner.name
  target_id = "lambda"
  arn       = aws_lambda_function.api.arn
  input     = jsonencode({ path = "/api/signals/scan", httpMethod = "GET" })
}

resource "aws_lambda_permission" "eventbridge" {
  statement_id  = "AllowEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.scanner.arn
}

# ============================================================================
# Outputs
# ============================================================================

output "frontend_url" {
  value       = "https://${aws_cloudfront_distribution.frontend.domain_name}"
  description = "Frontend CloudFront URL"
}

output "api_url" {
  value       = aws_apigatewayv2_api.main.api_endpoint
  description = "API Gateway URL"
}

output "s3_bucket" {
  value       = aws_s3_bucket.frontend.bucket
  description = "S3 bucket for frontend"
}

output "rds_endpoint" {
  value       = aws_db_instance.main.endpoint
  description = "RDS endpoint"
}
