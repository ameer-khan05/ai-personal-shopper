# AI Personal Shopper

An AI-powered shopping assistant that helps you find, compare, and choose products through natural conversation — built with the Anthropic Claude SDK.

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/ai-personal-shopper.git
cd ai-personal-shopper

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Open .env and add your real API keys

# 5. Run
python main.py
```

## Project Structure

```
ai-personal-shopper/
├── main.py                 # Application entry point
├── config.py               # Environment & app settings
├── requirements.txt        # Pinned Python dependencies
├── .env.example            # Template for required env vars
├── src/
│   ├── agents/             # Agent definitions & orchestration
│   ├── tools/              # Tool integrations (search, compare, …)
│   ├── memory/             # Persistent memory & preference storage
│   ├── models/             # Pydantic data models
│   └── ui/                 # Terminal UI components (Rich)
├── tests/                  # Test suite
├── config/                 # Additional configuration files
└── docs/                   # Documentation
```
