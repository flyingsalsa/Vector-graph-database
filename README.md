# Vector-Graph Database: Hybrid Semantic & Graph Search Engine

This project demonstrates how to combine **vector databases** (for semantic similarity search) and **graph databases** (for relationship modeling) to build advanced search and recommendation systems. It features practical examples using [Milvus](https://milvus.io/) for vector search and [Neo4j](https://neo4j.com/) for graph-based knowledge representation.

---

## Features

- **Semantic Search**: Uses [Sentence Transformers](https://www.sbert.net/) to generate embeddings and Milvus for fast similarity search.
- **Knowledge Graph**: Models entities, intents, and relationships using Neo4j.
- **Hybrid Search Engine**: Combines vector similarity with graph traversals for context-aware results.
- **Dockerized Setup**: Easy local deployment of Milvus and Neo4j using Docker Compose.

---

## Getting Started

### Prerequisites

- Docker & Docker Compose
- Python 3.6+
- pip

### 1. Start Milvus (Vector Database)

Create a `docker-compose.yml` file (see [docker-compose.yml](docker-compose.yml)) and run:

```sh
docker-compose up -d
```

### 2. Start Neo4j (Graph Database)

Create a `neo4j-docker-compose.yml` file (see [neo4j-docker-compose.yml](neo4j-docker-compose.yml)) and run:

```sh
docker-compose -f neo4j-docker-compose.yml up -d
```

### 3. Install Python Dependencies

```sh
pip install -r requirements.txt
```

Or, for notebook experimentation:

```sh
pip install pymilvus neo4j sentence-transformers numpy pandas
```

---

## Example Usage

### 1. Vector Search with Milvus

- Generate embeddings for documents using Sentence Transformers.
- Store and index vectors in Milvus.
- Perform semantic search to retrieve similar documents.

### 2. Knowledge Graph with Neo4j

- Model users, queries, documents, topics, and intents as nodes.
- Create relationships such as SEARCHED, HAS_INTENT, ABOUT, COVERS, RETURNED, VIEWED, BOOKMARKED.
- Query the graph for user interests, document topics, and more.

### 3. Hybrid Search Engine

- Combine vector search results with graph-based enrichment (e.g., related topics, documents).
- Example: Given a query, retrieve semantically similar documents and their graph context.

---

## Project Structure

```
.
├── docker-compose.yml
├── neo4j-docker-compose.yml
├── requirements.txt
├── src/
│   └── vec_graph_db/
│       ├── Hybrid_search.py
│       ├── milvus_collection_creation.py
│       ├── neo4j_graphDB_search.py
│       └── vectorDB_semantic_search.py
├── tests/
└── README.md
```

---

## Notebooks

WIP

---

## Key Technologies

- [Milvus](https://milvus.io/) - Open-source vector database
- [Neo4j](https://neo4j.com/) - Graph database
- [Sentence Transformers](https://www.sbert.net/) - Text embeddings
- [PyMilvus](https://pymilvus.io/) - Milvus Python SDK
- [Neo4j Python Driver](https://neo4j.com/docs/api/python-driver/current/) - Neo4j Python SDK

---

## Example: Hybrid Search Workflow

1. **Add Documents**: Store text in Milvus (vector DB) and as nodes in Neo4j (graph DB).
2. **Semantic Search**: Retrieve top-k similar documents using vector similarity.
3. **Graph Enrichment**: For each result, fetch related topics, user interactions, and connected documents from Neo4j.
4. **Result**: Return enriched, context-aware search results.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## References

- [Milvus Documentation](https://milvus.io/docs/)
- [Neo4j Documentation](https://neo4j.com/docs/)
- [Sentence Transformers](https://www.sbert.net/)
