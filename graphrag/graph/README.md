# Neo4jStorage

Store and load LangChain GraphDocument from Neo4j.

## Neo4j Setup

### Using conda (No sudo required)

```bash
# Install Java 21
conda install -y -c conda-forge openjdk=21

# Set JAVA_HOME
export JAVA_HOME=/data/fudawei/miniconda3/lib/jvm

# Start Neo4j
/data/fudawei/neo4j-community-2026.03.1/bin/neo4j start

# Check status
/data/fudawei/neo4j-community-2026.03.1/bin/neo4j status

# Stop Neo4j
/data/fudawei/neo4j-community-2026.03.1/bin/neo4j stop
```

### Using Docker

```bash
docker run -d --name neo4j \
  -p 7687:7687 \
  -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/Neo4j2024! \
  neo4j:latest
```

### First-time Setup

Neo4j requires changing the default password on first use:

```bash
# Stop Neo4j if running
/data/fudawei/neo4j-community-2026.03.1/bin/neo4j stop

# Clear data directory
rm -rf /data/fudawei/neo4j-community-2026.03.1/data/db/*

# Set initial password
export JAVA_HOME=/data/fudawei/miniconda3/lib/jvm
/data/fudawei/neo4j-community-2026.03.1/bin/neo4j-admin dbms set-initial-password Neo4j2024!

# Start Neo4j
/data/fudawei/neo4j-community-2026.03.1/bin/neo4j start
```

### Access Neo4j Browser

- URL: http://localhost:7474
- Username: `neo4j`
- Password: `Neo4j2024!`

## Usage

### Basic Usage

```python
from graphrag import get_neo4j_storage

# Create storage instance
storage = get_neo4j_storage(
    uri="bolt://localhost:7687",
    username="neo4j",
    password="Neo4j2024!"
)

# Save graph
storage.save_graph(graph_doc)

# Load graph
graph_doc = storage.load_graph()

# Get stats
stats = storage.stats()  # {'num_nodes': 100, 'num_relationships': 200}

# Execute custom Cypher query
results = storage.query("MATCH (n:Entity) RETURN n.id, n.node_type")

# Clear all data
storage.clear_graph()

# Close connection
storage.close()
```

### Using Context Manager

```python
from graphrag import get_neo4j_storage

with get_neo4j_storage() as storage:
    storage.save_graph(graph_doc)
    stats = storage.stats()
# Connection automatically closed
```

### Environment Variables

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=Neo4j2024!
```

```python
from graphrag import get_neo4j_storage

# Uses environment variables by default
storage = get_neo4j_storage()
```

## API Reference

| Method | Description |
|--------|-------------|
| `save_graph(graph_doc)` | Save GraphDocument to Neo4j |
| `load_graph()` | Load GraphDocument from Neo4j |
| `stats()` | Get node and relationship counts |
| `query(cypher, parameters)` | Execute custom Cypher query |
| `clear_graph()` | Delete all nodes and relationships |
| `close()` | Close the Neo4j driver |

## Running Tests

```bash
NEO4J_URI=bolt://localhost:7687 \
NEO4J_USERNAME=neo4j \
NEO4J_PASSWORD=Neo4j2024! \
uv run python3 -m graphrag.graph.graph_storage
```

If Neo4j is not accessible, unit tests will run automatically.

## Cypher Query Examples

```python
# Find all nodes with type "Person"
storage.query("MATCH (n:Entity) WHERE n.node_type = 'Person' RETURN n.id, n.age")

# Find relationships
storage.query("MATCH (a)-[r]->(b) RETURN a.id, type(r), b.id")

# Count nodes by type
storage.query("MATCH (n:Entity) RETURN n.node_type, count(n) as cnt")
```
