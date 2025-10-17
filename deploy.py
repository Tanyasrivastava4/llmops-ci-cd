# deploy.py
import os
from salad_cloud_sdk import SaladCloudSdk
#from salad_cloud_sdk.models import CreateContainerGroup
from salad_cloud_sdk.models import ContainerGroup 
# -------------------------------
# 1️⃣ Configuration
# -------------------------------
ORGANIZATION_NAME = os.getenv("ORGANIZATION_NAME", "intileo")  # replace or set in .env
PROJECT_NAME = os.getenv("PROJECT_NAME", "replit")             # replace or set in .env
CONTAINER_NAME = "rag-system-deploy"
IMAGE_NAME = "docker.io/tanyasrivastava930/rag_systems:latest"  # your Docker Hub image
GPU_TYPE = "RTX5090"

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
# 4️⃣ Create deployment request
# -------------------------------
request_body = ContainerGroup(
    name=CONTAINER_NAME,
    display_name=CONTAINER_NAME,
    container={
        "image": IMAGE_NAME,
        "resources": {
            "cpu": 2,
            "memory": 4096,          # in MB
            "gpu_classes": [GPU_TYPE]
        },
        "command": [
            "bash",
            "-c",
            "pip install -r /app/requirements.txt && python rag_systems/run_rag_pipeline.py"
        ],
        "environment_variables": {
            "HF_TOKEN": HF_TOKEN
        },
    },
    autostart_policy=True,      # ✅ Boolean as required
    restart_policy="always",    # ✅ String
    replicas=1
)

# -------------------------------
# 5️⃣ Deploy container group
# -------------------------------
try:
    result = sdk.container_groups.create_container_group(
        request_body=request_body,
        organization_name=ORGANIZATION_NAME,
        project_name=PROJECT_NAME
    )

    print("✅ Deployment request sent successfully!")
    print(f"🔗 Container Group ID: {result.id}")
    print(f"🌍 Status: {result.status}")

except Exception as e:
    print("❌ Deployment failed with error:")
    print(e)

print("🏁 Deployment script finished.")

