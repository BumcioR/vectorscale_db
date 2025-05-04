# Vector databases

# Similarity Search with TimescaleDB and pg_vectorscale

This project demonstrates how to build a similarity search system using TimescaleDB with the `pg_vectorscale` extension. It includes setting up a PostgreSQL database with vector support, creating vector indexes, embedding game descriptions, and performing vector similarity queries using SQLAlchemy and Python.

## Features

- Vector indexing with `pg_vectorscale`
- Embedding generation with Sentence Transformers
- SQLAlchemy ORM integration
- Vector similarity search (cosine distance)
- Dockerized PostgreSQL + TimescaleDB setup

## Requirements

- Docker + Docker Compose
- Python 3.11
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

## Installation

Clone the repository and navigate to the project directory:

```bash
git clone <repo_url>
cd vectorscale_db
```

Install Python dependencies using `uv`:

```bash
uv venv
source .venv/bin/activate
uv sync
```

This will create and activate a virtual environment and install all dependencies defined in `pyproject.toml`.

## Running the Database

Start the PostgreSQL container with TimescaleDB and pg_vectorscale:

```bash
docker compose up -d
```

It will run a container with TimescaleDB and automatically install the necessary extensions.

## Interacting with the Database

You can connect to the running database using a PostgreSQL client like `psql`:

```bash
psql -h localhost -p 5432 -U postgres -d similarity_search_service_db
```

Credentials:

    User: postgres

    Password: password

    Database: similarity_search_service_db

## Using the System

The system loads Steam games dataset, generates embeddings using `sentence-transformers` (`distiluse-base-multilingual-cased-v2`), inserts them into the database, and performs cosine similarity search based on textual queries.

Run the main script:

```bash
python main.py
```

## Dataset

The Steam Games dataset is loaded from HuggingFace:

- [https://huggingface.co/datasets/FronkonGames/steam-games-dataset](https://huggingface.co/datasets/FronkonGames/steam-games-dataset)

Only selected fields are used.

## License

MIT License
