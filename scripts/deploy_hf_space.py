# ==============================================================================
# scripts/deploy_hf_space.py
# ------------------------------------------------------------------------------
# Creates (if needed) and updates a Hugging Face Docker Space running this app.
#
#   python -m scripts.deploy_hf_space <hf-username>/<space-name> [--public]
#
# Prerequisite: log in once with `hf auth login` (stores your HF access token
# locally; this script never sees or handles it directly).
#
# What it does:
#   1. Creates the Space (Docker SDK, PRIVATE unless --public) if it doesn't
#      exist yet. The build instructions live in ./Dockerfile and the Space
#      settings in the YAML header of ./README.md.
#   2. Sets the Space's non-secret runtime variables from _SPACE_VARIABLES.
#   3. Uploads the project files. .env, local runtime data and caches are
#      excluded -- they never leave this machine.
#
# What it deliberately does NOT do: set GEMINI_API_KEY. Add it yourself as a
# *secret* under the Space's Settings -> "Variables and secrets", so the key is
# only ever entered into Hugging Face's own UI.
# ==============================================================================

import argparse

from huggingface_hub import HfApi

# Non-secret settings for the hosted app (same meaning as in .env.example).
# Change them later in the Space's Settings page -- no redeploy needed, just a
# restart of the Space.
_SPACE_VARIABLES = {
    "LLM_PROVIDER": "gemini",
    "GEMINI_CHAT_MODEL": "gemini-3.5-flash-lite",
    "EMBEDDING_PROVIDER": "local",
    "ENABLE_MULTI_QUERY": "true",
    "ENABLE_MULTI_HOP": "true",
}

# Never uploaded: secrets, local runtime data, caches, tooling folders.
_IGNORE_PATTERNS = [
    ".env",
    ".git/*",
    "vectorstore_db/*",
    "mlruns/*",
    "eval_results/*",
    "**/__pycache__/*",
    "*.pyc",
    ".pytest_cache/*",
    "venv/*",
    ".venv/*",
    "*.zip",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("space_id", help="e.g. your-hf-username/healthcare-insurance-assistant")
    parser.add_argument("--public", action="store_true", help="make the Space public (default: private)")
    args = parser.parse_args()

    api = HfApi()
    print(f"Logged in to Hugging Face as: {api.whoami()['name']}")

    api.create_repo(
        repo_id=args.space_id,
        repo_type="space",
        space_sdk="docker",
        private=not args.public,
        exist_ok=True,
    )
    for key, value in _SPACE_VARIABLES.items():
        api.add_space_variable(args.space_id, key, value)

    api.upload_folder(
        repo_id=args.space_id,
        repo_type="space",
        folder_path=".",
        ignore_patterns=_IGNORE_PATTERNS,
        commit_message="Deploy from local project",
    )

    print(f"Uploaded. Build progress: https://huggingface.co/spaces/{args.space_id}")
    print("If not done yet: add GEMINI_API_KEY as a SECRET in the Space's Settings -> Variables and secrets.")


if __name__ == "__main__":
    main()
