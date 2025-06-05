from vec_graph_db.vectorDB_semantic_search import SemanticSearchEngine
from vec_graph_db.neo4j_graphDB_search import KnowledgeGraph
from neo4j import GraphDatabase 

class HybridSearchEngine:
    def __init__(self, vector_model="all-MiniLM-L6-v2", vector_collection="hybrid_search",
                 neo4j_uri="bolt://localhost:7687", neo4j_user="neo4j", neo4j_password="password"):
        # Initialize vector search engine
        self.vector_search = SemanticSearchEngine(model_name=vector_model, collection_name=vector_collection)
        
        # Initialize knowledge graph
        self.graph = KnowledgeGraph(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
        
        # Map between Milvus IDs and Neo4j document IDs
        self.id_mapping = {}
    
    def add_document(self, text, properties=None):
        # Add to vector database
        vector_ids = self.vector_search.add_documents([text])
        vector_id = vector_ids[0]
        
        # Add to graph database
        props = properties or {}
        props["content"] = text
        props["vector_id"] = vector_id
        
        doc_name = props.get("name", f"Document-{vector_id}")
        doc_node = self.graph.create_entity("Document", doc_name, props)
        
        # Store mapping
        self.id_mapping[vector_id] = doc_node.id
        
        return vector_id, doc_node
    
    def add_topic(self, name, properties=None):
        return self.graph.create_entity("Topic", name, properties)
    
    def connect_document_to_topic(self, doc_name, topic_name, relationship_type="COVERS"):
        doc = self.graph.get_entity_by_name("Document", doc_name)
        topic = self.graph.get_entity_by_name("Topic", topic_name)
        
        if doc and topic:
            self.graph.create_relationship(doc, relationship_type, topic)
            return True
        return False
    
    def search(self, query, top_k=3, include_related=True):
        # Perform vector search
        vector_results = self.vector_search.search(query, top_k=top_k)
        
        if not include_related:
            return vector_results
        
        # Get document names from vector search results
        doc_contents = [result["text"] for result in vector_results]
        
        # Find these documents in the graph
        cypher_query = """
        MATCH (d:Document)
        WHERE d.content IN $contents
        OPTIONAL MATCH (d)-[:COVERS]->(t:Topic)
        OPTIONAL MATCH (t)<-[:COVERS]-(related:Document)
        WHERE NOT related.content IN $contents
        RETURN d.name as document, d.content as content, 
               collect(distinct t.name) as topics,
               collect(distinct related.content) as related_documents
        """
        
        graph_results = self.graph.run_query(cypher_query, {"contents": doc_contents})
        
        # Combine vector and graph results
        enriched_results = []
        for vec_result in vector_results:
            for graph_result in graph_results:
                if vec_result["text"] == graph_result["content"]:
                    enriched_results.append({
                        "document": graph_result["document"],
                        "content": graph_result["content"],
                        "score": vec_result["score"],
                        "topics": graph_result["topics"],
                        "related_documents": graph_result["related_documents"]
                    })
                    break
        
        return enriched_results

# Initialize the hybrid search engine
hybrid_search = HybridSearchEngine(vector_collection="hybrid_search_demo")

# Add documents with topics
docs = [
    {"text": "Python is a high-level programming language known for its readability and versatility.", 
     "name": "Python Introduction", "topics": ["Programming Languages", "Python"]},
    {"text": "JavaScript is primarily used for web development and runs in browsers.", 
     "name": "JavaScript Basics", "topics": ["Programming Languages", "JavaScript", "Web Development"]},
    {"text": "Machine learning models learn patterns from data without explicit programming.", 
     "name": "ML Fundamentals", "topics": ["Machine Learning", "Data Science"]},
    {"text": "Neural networks consist of layers of interconnected nodes or 'neurons'.", 
     "name": "Neural Network Architecture", "topics": ["Machine Learning", "Neural Networks"]},
    {"text": "Python is widely used in data science and machine learning applications.", 
     "name": "Python in Data Science", "topics": ["Python", "Data Science"]}
]

# Add documents and create topic connections
for doc in docs:
    # Add document
    _, doc_node = hybrid_search.add_document(doc["text"], {"name": doc["name"]})
    
    # Add topics and connect document to topics
    for topic_name in doc["topics"]:
        # Check if topic exists, create if not
        topic = hybrid_search.graph.get_entity_by_name("Topic", topic_name)
        if not topic:
            topic = hybrid_search.add_topic(topic_name)
        
        # Connect document to topic
        hybrid_search.graph.create_relationship(doc_node, "COVERS", topic)

print("Hybrid search engine populated with documents and topics")

# Perform a hybrid search
query = "How is Python used in machine learning?"
results = hybrid_search.search(query, top_k=2)

print(f"\nQuery: {query}")
for i, result in enumerate(results):
    print(f"\nResult {i+1}: {result['document']} (Score: {result['score']:.4f})")
    print(f"Content: {result['content']}")
    print(f"Topics: {', '.join(result['topics'])}")
    if result['related_documents']:
        print("Related documents:")
        for j, related in enumerate(result['related_documents']):
            print(f"  {j+1}. {related}")