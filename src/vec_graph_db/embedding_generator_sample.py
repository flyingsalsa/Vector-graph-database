from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

# Load a pre-trained Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample documents for our search engine
documents = [
    "Artificial intelligence is revolutionizing various industries.",
    "Machine learning is a subset of AI focused on learning from data.",
    "Neural networks are inspired by the human brain's structure.",
    "Deep learning uses multiple layers of neural networks for complex tasks.",
    "Natural Language Processing helps computers understand human language.",
    "Computer vision enables machines to interpret visual information.",
    "Reinforcement learning involves agents learning through trial and error.",
    "Data preprocessing is a crucial step in any machine learning pipeline.",
    "Feature engineering transforms raw data into useful model inputs.",
    "Model evaluation metrics help assess AI system performance."
]

# Create document IDs
doc_ids = list(range(len(documents)))

# Generate embeddings for all documents
embeddings = model.encode(documents)

print(f"Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
print(f"Sample embedding (first 5 values): {embeddings[0][:5]}")

### Milvus Setup
# Import Milvus client libraries
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection
)

# Connect to Milvus server
connections.connect("default", host="localhost", port="19530")

# Define collection name
collection_name = "document_search"

# Drop collection if it already exists
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

# Define the dimensionality of our embeddings
dim = embeddings.shape[1]  # 384 for 'all-MiniLM-L6-v2'

# Define collection schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500)
]
schema = CollectionSchema(fields=fields, description="Document search collection")

# Create collection
collection = Collection(name=collection_name, schema=schema)

# Create an index for vector field
index_params = {
    "metric_type": "COSINE",  # Similarity metric
    "index_type": "HNSW",     # Index type (Hierarchical Navigable Small World)
    "params": {"M": 8, "efConstruction": 64}  # HNSW index parameters
}
collection.create_index(field_name="embedding", index_params=index_params)

print(f"Created collection '{collection_name}' with vector dimension {dim}")