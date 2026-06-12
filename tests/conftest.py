import os

# Must be set before app modules are imported, since config.py instantiates
# Settings() at module level and anthropic_api_key has no default.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
