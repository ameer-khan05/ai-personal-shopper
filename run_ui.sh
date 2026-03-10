#!/usr/bin/env bash
# Launch the Streamlit web UI for AI Personal Shopper.
set -euo pipefail
cd "$(dirname "$0")"
source venv/bin/activate 2>/dev/null || true
streamlit run src/ui/app.py "$@"
