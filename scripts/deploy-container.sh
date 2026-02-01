#!/bin/bash
# Deploy RigaCap Backend as Container Image to Lambda
# This script builds a Docker image and pushes to ECR, then updates Lambda

set -e

AWS_REGION="us-east-1"
ECR_REPOSITORY="rigacap-prod-api"
LAMBDA_FUNCTION="rigacap-prod-api"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_TAG=$(git rev-parse --short HEAD 2>/dev/null || echo "latest")

echo "=============================================="
echo "RigaCap Backend Container Deployment"
echo "=============================================="
echo "AWS Account: ${AWS_ACCOUNT_ID}"
echo "ECR Registry: ${ECR_REGISTRY}"
echo "Image Tag: ${IMAGE_TAG}"
echo ""

# Step 0: Pre-deploy - Export current price data to S3
echo "💾 Step 0: Pre-deploy data export (preserving price history)..."
EXPORT_RESPONSE=$(curl -s -X POST "https://api.rigacap.com/api/data/pre-deploy" \
  -H "Content-Type: application/json" \
  -H "X-Admin-Key: ${ADMIN_API_KEY:-none}" \
  --max-time 120 2>/dev/null || echo '{"error": "API not reachable"}')

if echo "$EXPORT_RESPONSE" | grep -q '"success".*true'; then
  EXPORT_COUNT=$(echo "$EXPORT_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('count', 0))" 2>/dev/null || echo "0")
  echo "✅ Exported ${EXPORT_COUNT} symbols to S3 (data preserved)"
elif echo "$EXPORT_RESPONSE" | grep -q "API not reachable"; then
  echo "⚠️  Current Lambda not reachable - skipping pre-deploy export"
  echo "   (First deploy or Lambda is down - this is OK)"
else
  echo "⚠️  Export returned: ${EXPORT_RESPONSE}"
  echo "   Continuing with deploy (data may need to be re-fetched)"
fi
echo ""

# Step 1: Create ECR repository if it doesn't exist
echo "📦 Step 1: Checking ECR repository..."
aws ecr describe-repositories --repository-names ${ECR_REPOSITORY} --region ${AWS_REGION} 2>/dev/null || \
  aws ecr create-repository --repository-name ${ECR_REPOSITORY} --region ${AWS_REGION}
echo "✅ ECR repository ready"

# Step 2: Login to ECR
echo ""
echo "🔐 Step 2: Logging into ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REGISTRY}
echo "✅ Logged into ECR"

# Step 3: Build Docker image
echo ""
echo "🔨 Step 3: Building Docker image..."
cd "$(dirname "$0")/../backend"
# Build for x86_64 (Lambda default), with Lambda-compatible format
docker build -f Dockerfile.lambda -t ${ECR_REPOSITORY}:${IMAGE_TAG} \
  --platform linux/amd64 \
  --provenance=false \
  --sbom=false \
  .
docker tag ${ECR_REPOSITORY}:${IMAGE_TAG} ${ECR_REGISTRY}/${ECR_REPOSITORY}:${IMAGE_TAG}
docker tag ${ECR_REPOSITORY}:${IMAGE_TAG} ${ECR_REGISTRY}/${ECR_REPOSITORY}:latest
echo "✅ Docker image built"

# Step 4: Push to ECR
echo ""
echo "📤 Step 4: Pushing to ECR..."
docker push ${ECR_REGISTRY}/${ECR_REPOSITORY}:${IMAGE_TAG}
docker push ${ECR_REGISTRY}/${ECR_REPOSITORY}:latest
echo "✅ Image pushed to ECR"

# Step 5: Update Lambda function
echo ""
echo "🚀 Step 5: Updating Lambda function..."
aws lambda update-function-code \
  --function-name ${LAMBDA_FUNCTION} \
  --image-uri ${ECR_REGISTRY}/${ECR_REPOSITORY}:${IMAGE_TAG} \
  --region ${AWS_REGION}
echo "✅ Lambda updated"

# Step 6: Wait for Lambda to be ready
echo ""
echo "⏳ Step 6: Waiting for Lambda to be ready..."
aws lambda wait function-updated --function-name ${LAMBDA_FUNCTION} --region ${AWS_REGION}
echo "✅ Lambda is ready"

# Step 6.5: Sync DATABASE_URL with current RDS endpoint
echo ""
echo "🔄 Step 6.5: Syncing DATABASE_URL with RDS..."
RDS_ENDPOINT=$(aws rds describe-db-instances \
  --db-instance-identifier rigacap-prod-db-v2 \
  --region ${AWS_REGION} \
  --query "DBInstances[0].Endpoint.Address" \
  --output text 2>/dev/null || echo "")

if [ -n "$RDS_ENDPOINT" ] && [ "$RDS_ENDPOINT" != "None" ]; then
  # Get current environment variables
  CURRENT_ENV=$(aws lambda get-function-configuration \
    --function-name ${LAMBDA_FUNCTION} \
    --region ${AWS_REGION} \
    --query "Environment.Variables" \
    --output json)

  # Check if DATABASE_URL contains the correct endpoint
  CURRENT_DB_URL=$(echo "$CURRENT_ENV" | python3 -c "import sys,json; print(json.load(sys.stdin).get('DATABASE_URL',''))" 2>/dev/null || echo "")

  if [[ "$CURRENT_DB_URL" != *"$RDS_ENDPOINT"* ]]; then
    echo "⚠️  DATABASE_URL needs updating (current endpoint doesn't match RDS)"

    # Extract password from current URL or use placeholder
    DB_PASSWORD=$(echo "$CURRENT_DB_URL" | sed -n 's/.*:\/\/[^:]*:\([^@]*\)@.*/\1/p')
    if [ -z "$DB_PASSWORD" ]; then
      echo "❌ Could not extract database password from current config"
      echo "   Please run: terraform apply -var=\"db_password=YOUR_PASSWORD\""
    else
      # Build new environment with correct DATABASE_URL
      NEW_DB_URL="postgresql://rigacap:${DB_PASSWORD}@${RDS_ENDPOINT}:5432/rigacap"

      # Update all env vars, replacing DATABASE_URL
      NEW_ENV=$(echo "$CURRENT_ENV" | python3 -c "
import sys, json
env = json.load(sys.stdin)
env['DATABASE_URL'] = '${NEW_DB_URL}'
# Format for AWS CLI
pairs = ','.join([f'{k}={v}' for k,v in env.items()])
print(pairs)
")

      aws lambda update-function-configuration \
        --function-name ${LAMBDA_FUNCTION} \
        --region ${AWS_REGION} \
        --environment "Variables={${NEW_ENV}}" \
        --query "FunctionName" \
        --output text > /dev/null

      aws lambda wait function-updated --function-name ${LAMBDA_FUNCTION} --region ${AWS_REGION}
      echo "✅ DATABASE_URL updated to use ${RDS_ENDPOINT}"
    fi
  else
    echo "✅ DATABASE_URL already correct"
  fi
else
  echo "⚠️  RDS instance not found - skipping DATABASE_URL sync"
  echo "   Run 'terraform apply' to create the database"
fi

# Step 7: Test the deployment
echo ""
echo "🧪 Step 7: Testing deployment..."
curl -s https://api.rigacap.com/health | head -c 200
echo ""

# Step 8: Post-deploy warmup - reload data from S3
echo ""
echo "🔥 Step 8: Post-deploy warmup (loading data from S3)..."
WARMUP_RESPONSE=$(curl -s -X POST "https://api.rigacap.com/api/warmup" \
  --max-time 300 2>/dev/null || echo '{"error": "warmup timeout"}')

if echo "$WARMUP_RESPONSE" | grep -q '"status".*"warmed"'; then
  SYMBOLS=$(echo "$WARMUP_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('results',{}).get('price_data',{}).get('loaded',0))" 2>/dev/null || echo "?")
  echo "✅ Warmup complete: ${SYMBOLS} symbols loaded from S3"
else
  echo "⚠️  Warmup response: ${WARMUP_RESPONSE}"
  echo "   Data may need to be loaded manually via /api/data/load-batch"
fi

echo ""
echo "=============================================="
echo "✅ Deployment complete!"
echo "=============================================="
echo "Image: ${ECR_REGISTRY}/${ECR_REPOSITORY}:${IMAGE_TAG}"
echo "Lambda: ${LAMBDA_FUNCTION}"
echo ""
