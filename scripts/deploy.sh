#!/bin/bash
# RigaCap AWS Deployment Script
#
# This script deploys the full RigaCap application to AWS:
# 1. Builds the frontend React app
# 2. Packages the backend for Lambda
# 3. Applies Terraform infrastructure
# 4. Uploads frontend to S3
# 5. Invalidates CloudFront cache
#
# Prerequisites:
# - AWS CLI configured with credentials
# - Terraform installed
# - Node.js and npm installed
# - Python 3.11+ installed
#
# Usage:
#   ./scripts/deploy.sh                    # Full deploy
#   ./scripts/deploy.sh --frontend-only    # Only deploy frontend
#   ./scripts/deploy.sh --backend-only     # Only deploy backend
#   ./scripts/deploy.sh --plan             # Show Terraform plan without applying

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
BACKEND_DIR="$PROJECT_ROOT/backend"
TERRAFORM_DIR="$PROJECT_ROOT/infrastructure/terraform"

# Default options
DEPLOY_FRONTEND=true
DEPLOY_BACKEND=true
TERRAFORM_PLAN_ONLY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --frontend-only)
            DEPLOY_BACKEND=false
            shift
            ;;
        --backend-only)
            DEPLOY_FRONTEND=false
            shift
            ;;
        --plan)
            TERRAFORM_PLAN_ONLY=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Helper functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI not installed. Install with: brew install awscli"
        exit 1
    fi

    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured. Run: aws configure"
        exit 1
    fi

    # Check Terraform
    if ! command -v terraform &> /dev/null; then
        log_error "Terraform not installed. Install with: brew install terraform"
        exit 1
    fi

    # Check Node.js
    if ! command -v node &> /dev/null; then
        log_error "Node.js not installed. Install with: brew install node"
        exit 1
    fi

    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 not installed. Install with: brew install python"
        exit 1
    fi

    log_success "All prerequisites met"
}

build_frontend() {
    log_info "Building frontend..."
    cd "$FRONTEND_DIR"

    # Install dependencies
    npm install

    # Get API URL from Terraform output (if available)
    if [ -f "$TERRAFORM_DIR/.terraform/terraform.tfstate" ]; then
        API_URL=$(cd "$TERRAFORM_DIR" && terraform output -raw api_url 2>/dev/null || echo "")
        if [ -n "$API_URL" ]; then
            log_info "Setting API URL: $API_URL"
            export VITE_API_URL="$API_URL"
        fi
    fi

    # Build
    npm run build

    log_success "Frontend built successfully"
}

package_backend() {
    log_info "Packaging backend for Lambda..."
    cd "$BACKEND_DIR"

    # Clean previous package
    rm -rf package lambda.zip
    mkdir -p package

    # Install dependencies into package directory
    pip install -r requirements.txt -t package/ --quiet

    # Copy application code
    cp -r app package/
    cp main.py package/

    # Create Lambda handler wrapper
    cat > package/lambda_handler.py << 'EOF'
"""
AWS Lambda handler for FastAPI
"""
from mangum import Mangum
from main import app

handler = Mangum(app, lifespan="off")
EOF

    # Check if mangum is installed, if not add it
    if ! grep -q "mangum" requirements.txt; then
        pip install mangum -t package/ --quiet
    fi

    # Create zip
    cd package
    zip -r ../lambda.zip . -x "*.pyc" -x "__pycache__/*" -x "*.dist-info/*" --quiet
    cd ..

    # Move to terraform directory
    mv lambda.zip "$TERRAFORM_DIR/"

    # Clean up
    rm -rf package

    log_success "Backend packaged: $TERRAFORM_DIR/lambda.zip"
}

deploy_infrastructure() {
    log_info "Deploying infrastructure with Terraform..."
    cd "$TERRAFORM_DIR"

    # Initialize if needed
    if [ ! -d ".terraform" ]; then
        terraform init
    fi

    # Check for DB password
    if [ -z "$TF_VAR_db_password" ]; then
        log_warning "TF_VAR_db_password not set. Please enter database password:"
        read -s DB_PASSWORD
        export TF_VAR_db_password="$DB_PASSWORD"
    fi

    # Plan or Apply
    if [ "$TERRAFORM_PLAN_ONLY" = true ]; then
        terraform plan
        log_info "Plan complete. Run without --plan to apply."
    else
        terraform apply -auto-approve
        log_success "Infrastructure deployed"
    fi
}

upload_frontend() {
    log_info "Uploading frontend to S3..."
    cd "$TERRAFORM_DIR"

    # Get S3 bucket name
    S3_BUCKET=$(terraform output -raw s3_bucket)

    if [ -z "$S3_BUCKET" ]; then
        log_error "Could not get S3 bucket name from Terraform"
        exit 1
    fi

    # Sync frontend build to S3
    aws s3 sync "$FRONTEND_DIR/dist" "s3://$S3_BUCKET" --delete

    log_success "Frontend uploaded to s3://$S3_BUCKET"
}

invalidate_cloudfront() {
    log_info "Invalidating CloudFront cache..."
    cd "$TERRAFORM_DIR"

    # Get CloudFront distribution ID
    CF_DOMAIN=$(terraform output -raw frontend_url | sed 's|https://||')
    CF_DIST_ID=$(aws cloudfront list-distributions --query "DistributionList.Items[?DomainName=='$CF_DOMAIN'].Id" --output text)

    if [ -z "$CF_DIST_ID" ]; then
        log_warning "Could not find CloudFront distribution, skipping invalidation"
        return
    fi

    # Create invalidation
    aws cloudfront create-invalidation --distribution-id "$CF_DIST_ID" --paths "/*" > /dev/null

    log_success "CloudFront cache invalidated"
}

print_summary() {
    cd "$TERRAFORM_DIR"

    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}   Deployment Complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""

    if [ "$TERRAFORM_PLAN_ONLY" = false ]; then
        FRONTEND_URL=$(terraform output -raw frontend_url 2>/dev/null || echo "N/A")
        API_URL=$(terraform output -raw api_url 2>/dev/null || echo "N/A")

        echo -e "${BLUE}Frontend URL:${NC} $FRONTEND_URL"
        echo -e "${BLUE}API URL:${NC} $API_URL"
        echo ""
        echo -e "${YELLOW}Note: CloudFront may take a few minutes to propagate.${NC}"
    fi
}

# Main execution
main() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}   RigaCap AWS Deployment${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    check_prerequisites

    if [ "$DEPLOY_FRONTEND" = true ]; then
        build_frontend
    fi

    if [ "$DEPLOY_BACKEND" = true ]; then
        package_backend
        deploy_infrastructure
    fi

    if [ "$TERRAFORM_PLAN_ONLY" = false ]; then
        if [ "$DEPLOY_FRONTEND" = true ]; then
            upload_frontend
            invalidate_cloudfront
        fi
    fi

    print_summary
}

# Run main
main
