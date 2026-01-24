"""
Graph Retriever for Entity and Event-Edge Knowledge Graph

Implements GraphRAG augmentation:
1. Entity extraction from query
2. Graph traversal for related entities
3. Retrieval of chunks mentioning related entities
4. Event-edge awareness for temporal consistency

Addresses the "Long Context" problem by connecting scattered entity mentions.
"""

import logging
from typing import Optional, Set
from dataclasses import dataclass, field

import networkx as nx

from advanced_rag.config import AdvancedRAGConfig, DEFAULT_CONFIG
from advanced_rag.generation.schemas import EntityNode, EventEdge

logger = logging.getLogger(__name__)


@dataclass
class GraphSearchResult:
    """Result from graph-based entity search."""
    entity: str
    related_entities: list[str] = field(default_factory=list)
    events_involving: list[EventEdge] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    hop_distance: int = 0


class GraphRetriever:
    """
    Knowledge graph retriever for entity relationship traversal.
    
    Features:
    - Entity extraction from queries
    - Multi-hop graph traversal
    - Event-edge retrieval (who-did-what-to-whom)
    - Temporal filtering based on narrative position
    
    Why GraphRAG?
    - Claim: "Harry was raised by his aunt Petunia"
    - Vector search might not connect: Harry → Dursleys → Petunia
    - Graph: Harry --[relative_of]--> Petunia --[lives_with]--> Dursleys
    """
    
    def __init__(
        self,
        entity_graph: Optional[nx.DiGraph] = None,
        config: AdvancedRAGConfig = DEFAULT_CONFIG,
    ):
        self.graph = entity_graph or nx.DiGraph()
        self.config = config
        
        # Mapping from entity to chunk IDs
        self.entity_to_chunks: dict[str, set[str]] = {}
        
        # Event edges with temporal information
        self.events: list[EventEdge] = []
    
    def build_from_chunks(
        self,
        chunks: list[dict],
        entity_graph: Optional[nx.DiGraph] = None,
    ) -> None:
        """
        Build graph index from processed chunks.
        
        Args:
            chunks: List of chunk dicts with 'entities' and 'events'
            entity_graph: Optional pre-built entity graph
        """
        if entity_graph:
            self.graph = entity_graph
        
        # Build entity-to-chunk mapping
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "")
            entities = chunk.get("entities", [])
            events = chunk.get("events", [])
            
            # Map entities to chunks
            for entity in entities:
                name = entity if isinstance(entity, str) else entity.name
                if name not in self.entity_to_chunks:
                    self.entity_to_chunks[name] = set()
                self.entity_to_chunks[name].add(chunk_id)
                
                # Add to graph if not present
                if not self.graph.has_node(name):
                    self.graph.add_node(name)
            
            # Store events
            for event in events:
                if isinstance(event, dict):
                    event = EventEdge(**event)
                self.events.append(event)
                
                # Add event edge to graph
                self.graph.add_edge(
                    event.subject,
                    event.object,
                    action=event.action,
                    narrative_position=event.narrative_position,
                    chunk_id=event.source_chunk_id,
                )
        
        logger.info(f"Graph built: {self.graph.number_of_nodes()} nodes, "
                   f"{self.graph.number_of_edges()} edges, "
                   f"{len(self.events)} events")
    
    def find_entities_in_query(
        self,
        query: str,
        nlp=None,
    ) -> list[str]:
        """
        Extract entity mentions from query.
        
        Uses fuzzy matching against known entities.
        """
        query_lower = query.lower()
        found_entities = []
        
        # Simple substring matching against known entities
        for entity in self.graph.nodes():
            entity_lower = entity.lower()
            # Check if entity name or any word from entity appears in query
            if entity_lower in query_lower:
                found_entities.append(entity)
            else:
                # Check individual words for partial match
                entity_words = set(entity_lower.split())
                query_words = set(query_lower.split())
                if entity_words & query_words:  # Intersection
                    found_entities.append(entity)
        
        return found_entities
    
    def get_related_entities(
        self,
        entity: str,
        max_hops: int = 2,
        max_entities: int = 10,
    ) -> list[tuple[str, int]]:
        """
        Get related entities via graph traversal.
        
        Returns list of (entity_name, hop_distance) tuples.
        """
        if not self.graph.has_node(entity):
            return []
        
        related = []
        visited: Set[str] = {entity}
        
        # BFS traversal
        current_level = [entity]
        for hop in range(1, max_hops + 1):
            next_level = []
            for node in current_level:
                # Get neighbors (both directions)
                neighbors = set(self.graph.successors(node)) | set(self.graph.predecessors(node))
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        related.append((neighbor, hop))
                        next_level.append(neighbor)
                        
                        if len(related) >= max_entities:
                            return related
            
            current_level = next_level
        
        return related
    
    def get_events_involving(
        self,
        entity: str,
        narrative_range: Optional[tuple[float, float]] = None,
    ) -> list[EventEdge]:
        """
        Get events where entity is subject or object.
        
        Args:
            entity: Entity name
            narrative_range: Optional (start, end) to filter by position
        """
        entity_lower = entity.lower()
        matching_events = []
        
        for event in self.events:
            if (event.subject.lower() == entity_lower or 
                event.object.lower() == entity_lower):
                
                # Apply temporal filter if specified
                if narrative_range:
                    start, end = narrative_range
                    if not (start <= event.narrative_position <= end):
                        continue
                
                matching_events.append(event)
        
        # Sort by narrative position
        matching_events.sort(key=lambda e: e.narrative_position)
        return matching_events
    
    def get_chunks_for_entities(
        self,
        entities: list[str],
    ) -> set[str]:
        """Get chunk IDs that mention any of the given entities."""
        chunk_ids = set()
        for entity in entities:
            if entity in self.entity_to_chunks:
                chunk_ids.update(self.entity_to_chunks[entity])
        return chunk_ids
    
    def search(
        self,
        query: str,
        max_hops: int = None,
        max_results: int = 20,
        narrative_range: Optional[tuple[float, float]] = None,
    ) -> GraphSearchResult:
        """
        Full graph-based search for a query.
        
        Args:
            query: Search query
            max_hops: Maximum graph traversal hops
            max_results: Maximum chunk IDs to return
            narrative_range: Optional temporal filter
            
        Returns:
            GraphSearchResult with entities, events, and chunk IDs
        """
        max_hops = max_hops or self.config.graph.max_hop_distance
        
        # Extract entities from query
        query_entities = self.find_entities_in_query(query)
        
        if not query_entities:
            logger.debug(f"No entities found in query: {query[:50]}...")
            return GraphSearchResult(entity="", chunk_ids=[])
        
        logger.info(f"Found entities in query: {query_entities}")
        
        # Aggregate results from all query entities
        all_related = []
        all_events = []
        all_chunk_ids = set()
        
        for entity in query_entities:
            # Get related entities
            related = self.get_related_entities(entity, max_hops)
            all_related.extend(related)
            
            # Get events
            events = self.get_events_involving(entity, narrative_range)
            all_events.extend(events)
            
            # Get chunks for this entity and related
            entity_list = [entity] + [r[0] for r in related]
            chunks = self.get_chunks_for_entities(entity_list)
            all_chunk_ids.update(chunks)
        
        # Limit results
        chunk_ids_list = list(all_chunk_ids)[:max_results]
        
        return GraphSearchResult(
            entity=query_entities[0] if query_entities else "",
            related_entities=[r[0] for r in all_related[:10]],
            events_involving=all_events[:10],
            chunk_ids=chunk_ids_list,
            hop_distance=max_hops,
        )
    
    def get_entity_timeline(
        self,
        entity: str,
    ) -> list[dict]:
        """
        Get chronological timeline of events involving an entity.
        
        Useful for tracking character state changes over narrative.
        """
        events = self.get_events_involving(entity)
        
        timeline = []
        for event in events:
            timeline.append({
                "position": event.narrative_position,
                "chapter": event.chapter_index,
                "action": event.action,
                "with_entity": event.object if event.subject.lower() == entity.lower() else event.subject,
                "chunk_id": event.source_chunk_id,
            })
        
        return timeline
    
    def visualize(
        self,
        output_path: str = "entity_graph.html",
        max_nodes: int = 100,
    ) -> str:
        """
        Visualize the entity graph using pyvis.
        
        Returns path to saved HTML file.
        """
        try:
            from pyvis.network import Network
            
            # Create pyvis network
            net = Network(height="600px", width="100%", directed=True)
            
            # Add nodes (limit for performance)
            nodes = list(self.graph.nodes())[:max_nodes]
            for node in nodes:
                node_data = self.graph.nodes[node]
                color = {
                    "CHARACTER": "#ff6b6b",
                    "LOCATION": "#4ecdc4",
                    "ORGANIZATION": "#45b7d1",
                    "OBJECT": "#96ceb4",
                }.get(node_data.get("entity_type", ""), "#ffeaa7")
                
                net.add_node(node, label=node, color=color)
            
            # Add edges
            for u, v, data in self.graph.edges(data=True):
                if u in nodes and v in nodes:
                    action = data.get("action", "related")
                    net.add_edge(u, v, label=action)
            
            net.save_graph(output_path)
            logger.info(f"Graph visualization saved to {output_path}")
            return output_path
            
        except ImportError:
            logger.warning("pyvis not installed. Run: pip install pyvis")
            return ""
