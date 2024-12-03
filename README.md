# Patent Retrieval API

A FastAPI-based service for intelligent patent search and retrieval using multilingual sentence transformers.

## Features

- Semantic patent search using state-of-the-art language models
- Configurable precision-recall balance for search results
- RESTful API endpoints
- Health check monitoring

## Quick Start

### Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run main.py
```

# API Endpoints Documentation

## Search Patents
`POST /search`

Search for patents based on keywords with configurable precision-recall balance.

### Request Body
```json
{
    "keywords": ["artificial intelligence", "machine learning"],
    "precision_recall_balance": 0.5
}
```

# API Usage Examples
Using cURL:
```bash
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"keywords": ["artificial intelligence"], "precision_recall_balance": 0.5}'
```

Using Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/search",
    json={
        "keywords": ["artificial intelligence"],
        "precision_recall_balance": 0.5
    }
)
results = response.json()
```