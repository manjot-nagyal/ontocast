# Quick Start

This guide will help you get started with OntoCast quickly. We'll walk through a simple example of processing a document and viewing the results.

## Prerequisites

- OntoCast installed (see [Installation](installation.md))
- A sample document to process (e.g., a pdf or a markdown file)

## Basic Example

### Query the Server

```bash
curl -X POST http://url:port/process -F "file=@sample.pdf"

curl -X POST http://url:port/process -F "file=@sample.json"
```

`url` would be `localhost` for a locally running server, default port is 8999

### Running a Server

To start an OntoCast server:

```bash
serve \
    --working-directory WORKING_DIR \
    --ontology-directory ONTOLOGY_DIR \
    --logging-level info \
    --max-visits 2 \
    --clean
```

- `ONTOLOGY_DIR` is expected to contain ontologies in `turtle` format.
- `--max-visits` specifies the number of visits per decision node, e.g. `render_onto_triples` or `criticise_facts`
- for testing, you may use an optional parameter `--head-chunks` to process only `head_chunks` number of chunks
- LLM setting are provided via `.env`
- `--clean` (optional): If set, the triple store (Neo4j or Fuseki) will be initialized as clean (all data deleted on startup). Use with caution in production!

```sh
# Domain configuration (used for URI generation) 
CURRENT_DOMAIN=https://example.com
PORT=8999
LLM_TEMPERATURE=0.1

# openai flavor
# OpenAI API Key (required for LLM functionality)
LLM_PROVIDER=openai
OPENAI_API_KEY=your-api-key-here

# google gemini flavor
# Google API Key (required for LLM functionality)
LLM_PROVIDER=google
GOOGLE_API_KEY=your-google-api-key-here
LLM_MODEL_NAME=gemini-pro

# ollama flavor
# BASE URL (if using ollama)
LLM_BASE_URL=ollama-base-url
LLM_PROVIDER=ollama
LLM_MODEL_NAME=granite3.3```

### Receive Results

After processing, the ontology and the facts graph are returned in turtle format

```json
{
    "data": {
        "facts": "# facts in turtle format",
        "ontology": "# ontology in turtle format"
    }
  ...
}
```

## Next Steps

Now that you've processed your first document, you can:

1. Try processing different types of documents (PDF, Word)
2. Check the [API Reference](../reference/onto.md) for more details 