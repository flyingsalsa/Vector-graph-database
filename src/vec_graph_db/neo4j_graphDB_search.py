from neo4j import GraphDatabase
import uuid
### Creating a Knowledge Graph class for a Semantic Search System

class KnowledgeGraph:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def run_query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record for record in result]
    
    def clear_database(self):
        query = "MATCH (n) DETACH DELETE n"
        self.run_query(query)
        print("Database cleared")
    
    def create_entity(self, entity_type, name, properties=None):
        props = properties or {}
        props['name'] = name
        props['uuid'] = str(uuid.uuid4())  # Generate a unique identifier for the entity
        
        query = f"""
        CREATE (e:{entity_type} $props)
        RETURN e
        """
        result = self.run_query(query, {"props": props})
        return result[0]['e']
    
    # Older version of create_relationship (deprecated)
    # This method uses Neo4j's internal id() function, 
    # which is not recommended for long-term use.
    
    def create_relationship(self, from_entity, rel_type, to_entity, properties=None):
        props = properties or {}
        
        query = f"""
        MATCH (a), (b)
        WHERE a.uuid = $from_id AND b.uuid = $to_id
        CREATE (a)-[r:{rel_type} $props]->(b)
        RETURN r
        """
        result = self.run_query(query, {
            "from_id": from_entity["uuid"],
            "to_id": to_entity["uuid"],
            "props": props
        })
        return result[0]['r']
    
    def get_entity_by_name(self, entity_type, name):
        query = f"""
        MATCH (e:{entity_type} {{name: $name}})
        RETURN e
        """
        result = self.run_query(query, {"name": name})
        return result[0]['e'] if result else None
    
    def get_related_entities(self, entity, relationship_type=None):
        rel_clause = f"[r:{relationship_type}]" if relationship_type else "[r]"
        
        query = f"""
        MATCH (a)-{rel_clause}->(b)
        WHERE a.uuid = $entity_id
        RETURN b, type(r) as relationship_type, properties(r) as rel_properties
        """
        return self.run_query(query, {"entity_id": entity["uuid"]})
    
    def find_path_between_entities(self, from_name, from_type, to_name, to_type, max_depth=3):
        query = f"""
        MATCH path = shortestPath(
            (a:{from_type} {{name: $from_name}})-[*..{max_depth}]-(b:{to_type} {{name: $to_name}})
        )
        RETURN path
        """
        return self.run_query(query, {"from_name": from_name, "to_name": to_name})

# Connect to Neo4j
kg = KnowledgeGraph()

# Clear the database to start fresh
kg.clear_database()

### Adding Sample Entities and Relationships, normallly these would be loaded from a database or created dynamically
# Create entity types (nodes)
user1 = kg.create_entity("User", "John", {"age": 28, "occupation": "Software Engineer"})
user2 = kg.create_entity("User", "Alice", {"age": 35, "occupation": "Data Scientist"})

# Create documents
doc1 = kg.create_entity("Document", "Introduction to Machine Learning", {
    "content": "Machine learning is a subset of AI focused on learning from data.",
    "vector_id": 1  # Reference to the vector in Milvus
})

doc2 = kg.create_entity("Document", "Neural Networks Explained", {
    "content": "Neural networks are inspired by the human brain's structure.",
    "vector_id": 2
})

doc3 = kg.create_entity("Document", "Introduction to Deep Learning", {
    "content": "Deep learning uses multiple layers of neural networks for complex tasks.",
    "vector_id": 3
})

# Create topics
topic1 = kg.create_entity("Topic", "Machine Learning")
topic2 = kg.create_entity("Topic", "Neural Networks")
topic3 = kg.create_entity("Topic", "Deep Learning")

# Create intents
learn_intent = kg.create_entity("Intent", "Learning", {"description": "User wants to learn about a topic"})
compare_intent = kg.create_entity("Intent", "Comparison", {"description": "User wants to compare concepts"})

# Create queries
query1 = kg.create_entity("Query", "How do neural networks work?", {
    "timestamp": "2023-06-01T10:30:00", 
    "vector_id": 101  # Reference to query vector in Milvus
})

query2 = kg.create_entity("Query", "Difference between ML and deep learning?", {
    "timestamp": "2023-06-02T14:45:00",
    "vector_id": 102
})

# Create relationships
# User-Query relationships
kg.create_relationship(user1, "SEARCHED", query1, {"timestamp": "2023-06-01T10:30:00"})
kg.create_relationship(user2, "SEARCHED", query2, {"timestamp": "2023-06-02T14:45:00"})

# Query-Intent relationships
kg.create_relationship(query1, "HAS_INTENT", learn_intent)
kg.create_relationship(query2, "HAS_INTENT", compare_intent)

# Query-Topic relationships
kg.create_relationship(query1, "ABOUT", topic2)  # neural networks
kg.create_relationship(query2, "ABOUT", topic1)  # machine learning
kg.create_relationship(query2, "ABOUT", topic3)  # deep learning

# Document-Topic relationships
kg.create_relationship(doc1, "COVERS", topic1)  # ML document covers ML topic
kg.create_relationship(doc2, "COVERS", topic2)  # NN document covers NN topic
kg.create_relationship(doc3, "COVERS", topic3)  # DL document covers DL topic
kg.create_relationship(doc3, "REFERENCES", topic2)  # DL document references NN topic

# Query-Document relationships (based on search results)
kg.create_relationship(query1, "RETURNED", doc2, {"rank": 1, "score": 0.92})
kg.create_relationship(query1, "RETURNED", doc3, {"rank": 2, "score": 0.78})
kg.create_relationship(query2, "RETURNED", doc1, {"rank": 1, "score": 0.85})
kg.create_relationship(query2, "RETURNED", doc3, {"rank": 2, "score": 0.82})

# User-Document interactions
kg.create_relationship(user1, "VIEWED", doc2, {"timestamp": "2023-06-01T10:32:00", "duration": 120})
kg.create_relationship(user2, "VIEWED", doc1, {"timestamp": "2023-06-02T14:47:00", "duration": 90})
kg.create_relationship(user2, "BOOKMARKED", doc3, {"timestamp": "2023-06-02T15:10:00"})

print("Knowledge graph created successfully")


### Querying the Knowledge Graph for Enhanced Search

# Find documents about Neural Networks
query = """
MATCH (d:Document)-[:COVERS]->(t:Topic {name: 'Neural Networks'})
RETURN d.name as document, d.content as content
"""
results = kg.run_query(query)
print("Documents about Neural Networks:")
for record in results:
    print(f"- {record['document']}: {record['content']}")

# Find what topics a specific query was about
query = """
MATCH (q:Query {name: 'Difference between ML and deep learning?'})-[:ABOUT]->(t:Topic)
RETURN q.name as query, collect(t.name) as topics
"""
results = kg.run_query(query)
for record in results:
    print(f"\nQuery '{record['query']}' is about topics: {', '.join(record['topics'])}")

# Find documents returned for a specific intent
query = """
MATCH (q:Query)-[:HAS_INTENT]->(i:Intent {name: 'Learning'}),
      (q)-[r:RETURNED]->(d:Document)
RETURN i.name as intent, q.name as query, d.name as document, r.score as score
ORDER BY r.score DESC
"""
results = kg.run_query(query)
print("\nDocuments returned for 'Learning' intent:")
for record in results:
    print(f"- Query: '{record['query']}' → Document: '{record['document']}' (Score: {record['score']})")

# Find user search patterns
query = """
MATCH (u:User)-[:SEARCHED]->(q:Query)-[:ABOUT]->(t:Topic)
RETURN u.name as user, collect(distinct t.name) as topics_of_interest
"""
results = kg.run_query(query)
print("\nUser interests based on search queries:")
for record in results:
    print(f"- {record['user']} is interested in: {', '.join(record['topics_of_interest'])}")

# Find relationships between topics
query = """
MATCH (d:Document)-[:COVERS]->(t1:Topic),
      (d)-[:REFERENCES]->(t2:Topic)
WHERE t1 <> t2
RETURN t1.name as topic, t2.name as related_topic, d.name as connecting_document
"""
results = kg.run_query(query)
print("\nRelationships between topics:")
for record in results:
    print(f"- {record['topic']} is related to {record['related_topic']} via document '{record['connecting_document']}'")