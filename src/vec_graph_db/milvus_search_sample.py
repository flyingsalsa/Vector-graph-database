### Requires Milvus server running and pymilvus installed
# run " docker-compose up -d " to start Milvus server.
# see docker-compose.yml in the root directory for details.

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