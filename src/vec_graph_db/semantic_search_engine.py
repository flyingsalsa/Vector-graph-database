from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection
)

class SemanticSearchEngine:
    def __init__(self, model_name="all-MiniLM-L6-v2", collection_name="semantic_search"):
        # Load the sentence transformer model
        self.model = SentenceTransformer(model_name)
        self.collection_name = collection_name
        self.dim = self.model.get_sentence_embedding_dimension()
        
        # Connect to Milvus
        if not connections.has_connection("default"):
            connections.connect("default", host="localhost", port="19530")
        
        # Create collection if it doesn't exist
        self._initialize_collection()
    
    def _initialize_collection(self):
        # Drop existing collection if it exists
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
        
        # Define collection schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500)
        ]
        schema = CollectionSchema(fields=fields, description="Semantic search collection")
        
        # Create collection
        self.collection = Collection(name=self.collection_name, schema=schema)
        
        # Create an index for vector field
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64}
        }
        self.collection.create_index(field_name="embedding", index_params=index_params)
    
    def add_documents(self, documents):
        # Get current count
        current_count = self.collection.num_entities
        
        # Generate IDs
        doc_ids = list(range(current_count, current_count + len(documents)))
        
        # Generate embeddings
        embeddings = self.model.encode(documents).tolist()
        
        # Insert data
        insert_data = [doc_ids, embeddings, documents]
        self.collection.insert(insert_data)
        
        # Load collection to memory for search
        self.collection.load()
        
        return doc_ids
    
    def search(self, query, top_k=3):
        # Generate embedding for the query
        query_embedding = self.model.encode([query])[0].tolist()
        
        # Search parameters
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 32}
        }
        
        # Perform the search
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text"]
        )
        
        # Format results
        search_results = []
        for hit in results[0]:
            search_results.append({
                "text": hit.entity.get("text"),
                "score": hit.distance,
                "id": hit.id
            })
        
        return search_results

# Demo of the search engine class
search_engine = SemanticSearchEngine(collection_name="semantic_search_demo")

# Sample documents about various technologies
tech_documents = [
    "Blockchain is a distributed ledger technology enabling secure transactions.",
    "Cloud computing delivers computing services over the internet on-demand.",
    "Internet of Things (IoT) connects physical devices to the internet.",
    "Edge computing processes data near the source rather than in a centralized cloud.",
    "5G networks offer faster speeds and lower latency than previous generations.",
    "Quantum computing uses quantum mechanics to solve complex problems quickly.",
    "Augmented Reality (AR) overlays digital content onto the real world.",
    "Virtual Reality (VR) creates immersive digital environments for users.",
    "Cybersecurity protects systems and data from digital attacks.",
    "Big data analytics examines large datasets to uncover patterns and insights."
]

# Add documents to the search engine
doc_ids = search_engine.add_documents(tech_documents)
print(f"Added {len(doc_ids)} documents to the search engine")

# Perform a search
query = "How does cloud technology work?"
results = search_engine.search(query, top_k=3)

print(f"\nQuery: {query}")
for i, result in enumerate(results):
    print(f"Result {i+1}: {result['text']} (Score: {result['score']:.4f})")

# Interactive search terminal
def enhanced_interactive_search():
    """Enhanced interactive search with additional features"""
    print("=== Enhanced Interactive Search Terminal ===")
    print("Commands:")
    print("  - Enter any text to search")
    print("  - 'help' or 'h' for this help message")
    print("  - 'settings' or 's' to change search parameters")
    print("  - 'history' to view search history")
    print("  - 'quit', 'exit', or 'q' to stop")
    print("  - Ctrl+C to interrupt\n")
    
    search_history = []
    top_k = 3
    
    while True:
        try:
            query = input("Search> ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            elif query.lower() in ['help', 'h']:
                print("\nAvailable commands:")
                print("  help/h - Show this help")
                print("  settings/s - Change search parameters")
                print("  history - Show search history")
                print("  quit/exit/q - Exit program")
                continue
            elif query.lower() in ['settings', 's']:
                try:
                    new_k = input(f"Enter number of results (current: {top_k}): ").strip()
                    if new_k.isdigit():
                        top_k = int(new_k)
                        print(f"Updated to show {top_k} results")
                    else:
                        print("Invalid number, keeping current setting")
                except:
                    print("Invalid input, keeping current setting")
                continue
            elif query.lower() == 'history':
                if search_history:
                    print("\nSearch History:")
                    for i, hist_query in enumerate(search_history[-10:], 1):  # Show last 10
                        print(f"  {i}. {hist_query}")
                else:
                    print("No search history yet")
                continue
            elif query == '':
                continue
            
            # Add to history
            search_history.append(query)
            
            print(f"\nSearching for: '{query}'...")
            
            # Perform search
            results = search_engine.search(query, top_k=top_k)
            
            # Display results with better formatting
            print(f"\n{'='*60}")
            print(f"Results for: {query}")
            print(f"{'='*60}")
            
            if not results:
                print("No results found.")
            else:
                for i, result in enumerate(results, 1):
                    print(f"\n[Result {i}] (Score: {result['score']:.4f})")
                    print(f"{result['text']}")
                    if i < len(results):
                        print("-" * 40)
            
            print(f"\n{'='*60}\n")
            
        except KeyboardInterrupt:
            print("\n\nSearch interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    enhanced_interactive_search()