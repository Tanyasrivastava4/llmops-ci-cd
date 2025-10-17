# deploy.py
import os
from salad_cloud_sdk import SaladCloudSdk

# -------------------------------
# 1️⃣ Configuration
# -------------------------------
PROJECT_NAME = "replit"        # Your SaladCloud project
CONTAINER_NAME = "llmops-container"
IMAGE_NAME = "tanyasrivastava930/llmops-rag:latest"

# -------------------------------
# 2️⃣ Environment Variables
# -------------------------------
SALAD_API_KEY = os.getenv("SALAD_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN", "")

if not SALAD_API_KEY:
    raise ValueError("❌ Missing SALAD_API_KEY in environment variables.")

# -------------------------------
# 3️⃣ Initialize SDK
# -------------------------------
sdk = SaladCloudSdk(api_key=SALAD_API_KEY)

print("🚀 Deploying container group to SaladCloud...")

# -------------------------------
# 4️⃣ Prepare deployment request
# -------------------------------
container_group_request = {
    "name": CONTAINER_NAME,
    "container": {
        "image": IMAGE_NAME,
        "resources": {
            "gpu_type": "RTX5090",
            "cpu": 2,
            "memory": 4096   # memory in MB
        },
        "command": [
            "bash",
            "-c",
            "pip install -r /app/requirements.txt && python rag_systems/run_rag_pipeline.py"
        ],
        "environment_variables": {
            "HF_TOKEN": HF_TOKEN
        }
    },
    "autostart_policy": "always",       # lowercase string
    "replicas": 1,
    "restart_policy": "on_failure"      # lowercase string
}

# -------------------------------
# 5️⃣ Deploy container group
# -------------------------------
try:
    result = sdk.container_groups.create_container_group(
        project_name=PROJECT_NAME,
        request_body=container_group_request
    )
    print("✅ Deployment request sent successfully!")
    print(f"🔗 Container Group ID: {result.id}")
    print(f"🌍 Status: {result.status}")

except Exception as e:
    print("❌ Deployment failed with error:")
    print(e)

print("🏁 Deployment script finished.")

