import os
from salad_cloud_sdk import SaladCloudSdk
from dotenv import load_dotenv

# ---------------------------------------
# Load environment variables
# ---------------------------------------
load_dotenv()

SALAD_API_KEY = os.getenv("SALAD_API_KEY", "")
ORGANIZATION_NAME = os.getenv("ORGANIZATION_NAME", "")
PROJECT_NAME = os.getenv("PROJECT_NAME", "")
IMAGE_NAME = os.getenv("IMAGE_NAME", "docker.io/yourusername/llmops:latest")
CONTAINER_NAME = os.getenv("CONTAINER_NAME", "rag-llmops-deploy")
GPU_TYPE = os.getenv("GPU_TYPE", "ed563892-aacd-40f5-80b7-90c9be6c759b")  # Example GPU class
HF_TOKEN = os.getenv("HF_TOKEN", "")

# ---------------------------------------
# Initialize SDK
# ---------------------------------------
sdk = SaladCloudSdk(api_key=SALAD_API_KEY, timeout=10000)

print("🚀 Deploying container group to SaladCloud...")

# ---------------------------------------
# Define container group request
# ---------------------------------------
container_group_request = {
    "name": CONTAINER_NAME,
    "display_name": CONTAINER_NAME,
    "container": {
        "image": IMAGE_NAME,
        "resources": {
            "cpu": 2,
            "memory": 4096,  # MB
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
    "autostart_policy": True,
    "restart_policy": "always",
    "replicas": 1,
    "country_codes": ["us"],
    "priority": "high"
}

# ---------------------------------------
# Create container group
# ---------------------------------------
try:
    result = sdk.container_groups.create_container_group(
        organization_name=ORGANIZATION_NAME,
        project_name=PROJECT_NAME,
        request_body=container_group_request
    )
    print("✅ Deployment successful!")
    print(result)
except Exception as e:
    print(f"❌ Deployment failed with error:\n{e}")

