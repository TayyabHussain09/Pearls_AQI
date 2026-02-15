"""Test Hopsworks connection"""
import sys
sys.path.insert(0, '.')

import hopsworks
from config.settings import settings
import logging

logging.basicConfig(level=logging.INFO)

print("="*50)
print("Testing Hopsworks Connection")
print("="*50)

# Login
print("1. Logging in to Hopsworks...")
conn = hopsworks.login(
    host=settings.HOPSWORKS_HOST,
    project=settings.HOPSWORKS_PROJECT_NAME,
    api_key_value=settings.HOPSWORKS_API_KEY
)
print(f"   Connected to project: {conn.name}")

# Get feature store
print("2. Getting feature store...")
fs = conn.get_feature_store()
print(f"   Feature Store: {fs.name}")

# Get model registry
print("3. Getting model registry...")
mr = conn.get_model_registry()
print(f"   Model Registry: {mr}")

print("="*50)
print("SUCCESS - All connections working!")
print("="*50)
