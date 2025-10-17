#!/bin/bash
set -e  # Exit on first error

echo "======================================"
echo "🚀 Starting SaladCloud Deployment..."
echo "======================================"

# --------------------------------------
# 1️⃣ Install dependencies
# --------------------------------------
echo "📦 Installing Python dependencies..."
pip install --upgrade pip >/dev/null 2>&1
pip install salad-cloud-sdk python-dotenv >/dev/null 2>&1

# --------------------------------------
# 2️⃣ Environment variables check
# --------------------------------------
if [ -z "$SALAD_API_KEY" ]; then
  echo "❌ ERROR: SALAD_API_KEY not set. Please add it as an environment variable."
  exit 1
fi

if [ -z "$ORGANIZATION_NAME" ]; then
  export ORGANIZATION_NAME="intileo"
fi

if [ -z "$PROJECT_NAME" ]; then
  export PROJECT_NAME="replit"
fi

# Hugging Face token is optional
if [ -z "$HF_TOKEN" ]; then
  echo "⚠️ Warning: HF_TOKEN not found. Continuing without it..."
else
  echo "🔑 Hugging Face token detected."
fi

# --------------------------------------
# 3️⃣ Run deployment script
# --------------------------------------
echo "🚧 Running deploy.py ..."
python deploy.py

# --------------------------------------
# 4️⃣ Post-deployment status
# --------------------------------------
echo "✅ Deployment triggered successfully!"
echo "🌍 Check your SaladCloud dashboard for container status."
echo "======================================"
echo "🏁 Deployment completed."
echo "======================================"

