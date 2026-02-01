# RigaCap AWS Infrastructure - Terraform
#
# This creates:
# - S3 bucket for frontend (static website)
# - CloudFront distribution (CDN) with SSL
# - ACM Certificate for rigacap.com
# - Lambda function (backend API)
# - API Gateway (REST API) with custom domain
# - RDS PostgreSQL (database)
# - EventBridge (scheduled scans)
# - Route53 DNS (optional)

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

# ACM certificates for CloudFront MUST be in us-east-1
provider "aws" {
  alias  = "us_east_1"
  region = "us-east-1"
}

# ============================================================================
# Variables
# ============================================================================

variable "aws_region" {
  default = "us-east-1"
}

variable "project_name" {
  default = "rigacap"
}

variable "environment" {
  default = "prod"
}

variable "domain_name" {
  description = "Primary domain name"
  default     = "rigacap.com"
}

variable "use_route53" {
  description = "Whether to create Route53 hosted zone (set false if using external DNS)"
  default     = true
}

variable "db_password" {
  description = "RDS database password"
  sensitive   = true
}

variable "jwt_secret_key" {
  description = "Secret key for JWT tokens"
  sensitive   = true
}

variable "stripe_secret_key" {
  description = "Stripe secret key"
  sensitive   = true
  default     = ""
}

variable "stripe_webhook_secret" {
  description = "Stripe webhook signing secret"
  sensitive   = true
  default     = ""
}

variable "stripe_price_id" {
  description = "Stripe price ID for monthly subscription"
  default     = ""
}

variable "stripe_price_id_annual" {
  description = "Stripe price ID for annual subscription"
  default     = ""
}

variable "turnstile_secret_key" {
  description = "Cloudflare Turnstile secret key"
  sensitive   = true
  default     = ""
}

variable "smtp_user" {
  description = "Gmail address for SMTP (e.g., your-email@gmail.com)"
  default     = ""
}

variable "smtp_pass" {
  description = "Gmail App Password for SMTP (16 characters, no spaces)"
  sensitive   = true
  default     = ""
}

variable "lambda_image_tag" {
  description = "Docker image tag for Lambda container"
  default     = "latest"
}

locals {
  prefix = "${var.project_name}-${var.environment}"
}

# ============================================================================
# ACM Certificate (FREE SSL from AWS)
# ============================================================================

# Certificate for rigacap.com and *.rigacap.com
resource "aws_acm_certificate" "main" {
  provider          = aws.us_east_1  # Must be us-east-1 for CloudFront
  domain_name       = var.domain_name
  subject_alternative_names = ["*.${var.domain_name}"]
  validation_method = "DNS"

  lifecycle {
    create_before_destroy = true
  }

  tags = {
    Name = "${local.prefix}-cert"
  }
}

# Route53 Hosted Zone (optional - skip if using external DNS)
resource "aws_route53_zone" "main" {
  count = var.use_route53 ? 1 : 0
  name  = var.domain_name

  tags = {
    Name = "${local.prefix}-zone"
  }
}

# DNS validation records (only if using Route53)
resource "aws_route53_record" "cert_validation" {
  for_each = var.use_route53 ? {
    for dvo in aws_acm_certificate.main.domain_validation_options : dvo.domain_name => {
      name   = dvo.resource_record_name
      record = dvo.resource_record_value
      type   = dvo.resource_record_type
    }
  } : {}

  allow_overwrite = true
  name            = each.value.name
  records         = [each.value.record]
  ttl             = 60
  type            = each.value.type
  zone_id         = aws_route53_zone.main[0].zone_id
}

# Wait for certificate validation
resource "aws_acm_certificate_validation" "main" {
  provider                = aws.us_east_1
  certificate_arn         = aws_acm_certificate.main.arn
  validation_record_fqdns = var.use_route53 ? [for record in aws_route53_record.cert_validation : record.fqdn] : []

  # If not using Route53, you'll need to manually add DNS records
  # and this will wait until they're validated
}

# ============================================================================
# Route53 DNS Records (only if using Route53)
# ============================================================================

# A record for rigacap.com -> CloudFront
resource "aws_route53_record" "frontend" {
  count   = var.use_route53 ? 1 : 0
  zone_id = aws_route53_zone.main[0].zone_id
  name    = var.domain_name
  type    = "A"

  alias {
    name                   = aws_cloudfront_distribution.frontend.domain_name
    zone_id                = aws_cloudfront_distribution.frontend.hosted_zone_id
    evaluate_target_health = false
  }
}

# A record for www.rigacap.com -> CloudFront
resource "aws_route53_record" "frontend_www" {
  count   = var.use_route53 ? 1 : 0
  zone_id = aws_route53_zone.main[0].zone_id
  name    = "www.${var.domain_name}"
  type    = "A"

  alias {
    name                   = aws_cloudfront_distribution.frontend.domain_name
    zone_id                = aws_cloudfront_distribution.frontend.hosted_zone_id
    evaluate_target_health = false
  }
}

# A record for api.rigacap.com -> API Gateway
resource "aws_route53_record" "api" {
  count   = var.use_route53 ? 1 : 0
  zone_id = aws_route53_zone.main[0].zone_id
  name    = "api.${var.domain_name}"
  type    = "A"

  alias {
    name                   = aws_apigatewayv2_domain_name.api.domain_name_configuration[0].target_domain_name
    zone_id                = aws_apigatewayv2_domain_name.api.domain_name_configuration[0].hosted_zone_id
    evaluate_target_health = false
  }
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
# CloudFront Distribution (with SSL)
# ============================================================================

resource "aws_cloudfront_distribution" "frontend" {
  enabled             = true
  default_root_object = "index.html"
  price_class         = "PriceClass_100"

  # Custom domain names
  aliases = [var.domain_name, "www.${var.domain_name}"]

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

  # SSL Certificate from ACM
  viewer_certificate {
    acm_certificate_arn      = aws_acm_certificate_validation.main.certificate_arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }

  tags = {
    Name = "${local.prefix}-frontend"
  }

  depends_on = [aws_acm_certificate_validation.main]
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

# S3 bucket for Lambda code (kept for backwards compatibility)
resource "aws_s3_bucket" "lambda_code" {
  bucket = "${local.prefix}-lambda-deploy-774558858301"
}

# ============================================================================
# ECR Repository - Container Image for Lambda (10GB limit vs 250MB for zip)
# ============================================================================

resource "aws_ecr_repository" "api" {
  name                 = "${local.prefix}-api"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Name = "${local.prefix}-api"
  }
}

# Lifecycle policy to keep only last 5 images
resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 5 images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 5
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

# S3 bucket for persistent price data (parquet files)
resource "aws_s3_bucket" "price_data" {
  bucket = "${local.prefix}-price-data-774558858301"

  tags = {
    Name = "${local.prefix}-price-data"
  }
}

# IAM policy for Lambda to access price data bucket
resource "aws_iam_role_policy" "lambda_s3_price_data" {
  name = "${local.prefix}-lambda-s3-price-data"
  role = aws_iam_role.lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.price_data.arn,
          "${aws_s3_bucket.price_data.arn}/*"
        ]
      }
    ]
  })
}

# Lambda Function - Using Container Image (10GB limit instead of 250MB)
resource "aws_lambda_function" "api" {
  function_name = "${local.prefix}-api"
  role          = aws_iam_role.lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.api.repository_url}:${var.lambda_image_tag}"
  timeout       = 900  # 15 minutes (max for Lambda)
  memory_size   = 3008  # Max memory for faster cold starts and large data loads

  environment {
    variables = {
      DATABASE_URL          = "postgresql://${aws_db_instance.main.username}:${var.db_password}@${aws_db_instance.main.endpoint}/${aws_db_instance.main.db_name}"
      ENVIRONMENT           = var.environment
      FRONTEND_URL          = "https://${var.domain_name}"
      JWT_SECRET_KEY        = var.jwt_secret_key
      STRIPE_SECRET_KEY     = var.stripe_secret_key
      STRIPE_WEBHOOK_SECRET = var.stripe_webhook_secret
      STRIPE_PRICE_ID       = var.stripe_price_id
      STRIPE_PRICE_ID_ANNUAL = var.stripe_price_id_annual
      TURNSTILE_SECRET_KEY  = var.turnstile_secret_key
      PRICE_DATA_BUCKET     = aws_s3_bucket.price_data.bucket
      SMTP_HOST             = "smtp.gmail.com"
      SMTP_PORT             = "587"
      SMTP_USER             = var.smtp_user
      SMTP_PASS             = var.smtp_pass
      FROM_EMAIL            = var.smtp_user  # Use same as SMTP_USER
      FROM_NAME             = "RigaCap Signals"
    }
  }

  tags = {
    Name = "${local.prefix}-api"
  }

  depends_on = [aws_ecr_repository.api]
}

# ============================================================================
# API Gateway
# ============================================================================

resource "aws_apigatewayv2_api" "main" {
  name          = "${local.prefix}-api"
  protocol_type = "HTTP"

  cors_configuration {
    allow_origins = [
      "https://${var.domain_name}",
      "https://www.${var.domain_name}",
      "http://localhost:3000",
      "http://localhost:5173",
      "http://localhost:5176"
    ]
    allow_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    allow_headers = ["*"]
    allow_credentials = true
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

# Custom domain for API (api.rigacap.com)
resource "aws_apigatewayv2_domain_name" "api" {
  domain_name = "api.${var.domain_name}"

  domain_name_configuration {
    certificate_arn = aws_acm_certificate_validation.main.certificate_arn
    endpoint_type   = "REGIONAL"
    security_policy = "TLS_1_2"
  }

  depends_on = [aws_acm_certificate_validation.main]
}

# Map custom domain to API Gateway stage
resource "aws_apigatewayv2_api_mapping" "api" {
  api_id      = aws_apigatewayv2_api.main.id
  domain_name = aws_apigatewayv2_domain_name.api.id
  stage       = aws_apigatewayv2_stage.default.id
}

# ============================================================================
# RDS PostgreSQL
# ============================================================================

# Security group for RDS - allows Lambda access
resource "aws_security_group" "rds" {
  name        = "${local.prefix}-rds-sg-v2"
  description = "Security group for RDS PostgreSQL"

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # TODO: Restrict to Lambda in production
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${local.prefix}-rds-sg"
  }
}

resource "aws_db_instance" "main" {
  identifier              = "${local.prefix}-db-v2"
  engine                  = "postgres"
  engine_version          = "15"
  instance_class          = "db.t3.micro"
  allocated_storage       = 20
  storage_type            = "gp2"
  db_name                 = "rigacap"
  username                = "rigacap"
  password                = var.db_password
  publicly_accessible     = true
  skip_final_snapshot     = true
  deletion_protection     = true  # Prevent accidental deletion
  vpc_security_group_ids  = [aws_security_group.rds.id]

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
# EventBridge - Lambda Warmer (keeps Lambda warm to avoid cold starts)
# ============================================================================

resource "aws_cloudwatch_event_rule" "warmer" {
  name                = "${local.prefix}-warmer"
  description         = "Keep Lambda warm by pinging it every 5 minutes"
  schedule_expression = "rate(5 minutes)"
}

resource "aws_cloudwatch_event_target" "warmer" {
  rule      = aws_cloudwatch_event_rule.warmer.name
  target_id = "lambda-warmer"
  arn       = aws_lambda_function.api.arn
  input     = jsonencode({ path = "/health", httpMethod = "GET", warmer = true })
}

resource "aws_lambda_permission" "warmer" {
  statement_id  = "AllowWarmerEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.warmer.arn
}

# ============================================================================
# Outputs
# ============================================================================

output "frontend_url" {
  value       = "https://${var.domain_name}"
  description = "Frontend URL (custom domain)"
}

output "frontend_cloudfront_url" {
  value       = "https://${aws_cloudfront_distribution.frontend.domain_name}"
  description = "Frontend CloudFront URL (for testing)"
}

output "api_url" {
  value       = "https://api.${var.domain_name}"
  description = "API URL (custom domain)"
}

output "api_gateway_url" {
  value       = aws_apigatewayv2_api.main.api_endpoint
  description = "API Gateway URL (for testing)"
}

output "s3_bucket" {
  value       = aws_s3_bucket.frontend.bucket
  description = "S3 bucket for frontend"
}

output "rds_endpoint" {
  value       = aws_db_instance.main.endpoint
  description = "RDS endpoint"
}

output "nameservers" {
  value       = var.use_route53 ? aws_route53_zone.main[0].name_servers : []
  description = "Route53 nameservers (point your domain registrar to these)"
}

output "cloudfront_domain" {
  value       = aws_cloudfront_distribution.frontend.domain_name
  description = "CloudFront domain (for CNAME if not using Route53)"
}

output "api_gateway_domain" {
  value       = aws_apigatewayv2_domain_name.api.domain_name_configuration[0].target_domain_name
  description = "API Gateway domain (for CNAME if not using Route53)"
}

output "price_data_bucket" {
  value       = aws_s3_bucket.price_data.bucket
  description = "S3 bucket for persistent price data"
}

output "ecr_repository_url" {
  value       = aws_ecr_repository.api.repository_url
  description = "ECR repository URL for Lambda container image"
}

# ============================================================================
# DNS Setup Instructions
# ============================================================================
# After deployment, update your domain's nameservers to the values in 'nameservers' output.
# Or if using external DNS, create CNAME records pointing to cloudfront_domain and api_gateway_domain.
