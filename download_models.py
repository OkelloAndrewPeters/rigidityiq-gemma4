# ── download_models.py ───────────────────────────────────────────────────────
# One-time setup script for RigidityIQ.
#
# Run this script ONCE at a location with internet access — a regional hub,
# an NGO office, or any machine with a stable connection. After it completes,
# the application runs fully offline on that device indefinitely.
#
# This script implements the "Hub-and-Spoke" deployment model described in the
# project writeup: provision once at the centre, deploy to the field via USB.
#
# What this script downloads:
#   - all-MiniLM-L6-v2 (~80MB): the sentence-transformer embedding model
#     used by ChromaDB for semantic retrieval of clinical guidelines.
#
# Note: The Gemma 4 E2B model (~7.2GB) is downloaded separately via Ollama:
#   ollama pull gemma4:e2b
# Both must be downloaded before running app.py for the first time.
# ────────────────────────────────────────────────────────────────────────────

from sentence_transformers import SentenceTransformer

print("Downloading model to local cache...")
print("This is a one-time download (~80MB). Ensure you have internet access.")

# cache_folder="./models" stores the model in the application directory
# rather than the system-default HuggingFace cache. This is intentional:
# it keeps the model co-located with the application for portable deployment
# (e.g. the entire folder can be copied to a USB drive or a new device),
# and it avoids symlink issues on Windows where the default cache uses
# symlinks that require Developer Mode to be enabled.
model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="./models")
print("Done. You can now run offline.")
