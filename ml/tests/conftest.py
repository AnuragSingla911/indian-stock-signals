import os

# Force deterministic offline data for all tests (no network, reproducible).
os.environ["ISS_OFFLINE"] = "1"
