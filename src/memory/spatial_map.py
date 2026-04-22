"""Spatial map with topological graph and A* path planning."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
import heapq

# Type alias for 2D position (x, z) in Minecraft coordinates
Position = Tuple[float, float]


@dataclass
class MapNode:
    """A node in the topological map representing a navigable location.

    Attributes:
        position: The (x, z) coordinates of this node.
        objects_seen: Set of object types observed at or near this location.
        visited_count: Number of times the agent has visited this node.
        last_visited: Timestamp of the most recent visit.
    """

    position: Position
    objects_seen: Set[str] = field(default_factory=set)
    visited_count: int = 0
    last_visited: Optional[datetime] = None

    def add_object(self, object_type: str) -> None:
        """Add an object type to the set of objects seen at this node."""
        self.objects_seen.add(object_type)

    def mark_visited(self) -> None:
        """Mark this node as visited, incrementing count and updating timestamp."""
        self.visited_count += 1
        self.last_visited = datetime.now()


def euclidean_distance(p1: Position, p2: Position) -> float:
    """Calculate the Euclidean distance between two positions.

    Args:
        p1: First position (x, z).
        p2: Second position (x, z).

    Returns:
        The Euclidean distance between the two positions.
    """
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


class TopologicalMap:
    """A topological map for spatial navigation using a graph structure.

    The map consists of nodes (navigable positions) and edges (traversable
    connections between nodes). Supports A* path finding between nodes.

    Attributes:
        nodes: Dictionary mapping positions to MapNode objects.
        edges: Set of bidirectional edges as tuples of positions.
    """

    def __init__(self) -> None:
        """Initialize an empty topological map."""
        self.nodes: Dict[Position, MapNode] = {}
        self.edges: Set[Tuple[Position, Position]] = set()

    def add_node(self, position: Position) -> MapNode:
        """Add a node at the given position.

        If a node already exists at this position, returns the existing node.

        Args:
            position: The (x, z) coordinates for the new node.

        Returns:
            The MapNode at the specified position.
        """
        if position not in self.nodes:
            self.nodes[position] = MapNode(position=position)
        return self.nodes[position]

    def add_edge(self, p1: Position, p2: Position) -> None:
        """Add a bidirectional edge between two positions.

        Args:
            p1: First position.
            p2: Second position.
        """
        # Store edges in a canonical form (sorted) to ensure bidirectionality
        edge = (min(p1, p2), max(p1, p2))
        self.edges.add(edge)

    def has_edge(self, p1: Position, p2: Position) -> bool:
        """Check if an edge exists between two positions.

        Args:
            p1: First position.
            p2: Second position.

        Returns:
            True if an edge exists between p1 and p2, False otherwise.
        """
        edge = (min(p1, p2), max(p1, p2))
        return edge in self.edges

    def get_neighbors(self, position: Position) -> List[Position]:
        """Get all neighboring positions connected by edges.

        Args:
            position: The position to find neighbors for.

        Returns:
            List of neighboring positions.
        """
        neighbors = []
        for p1, p2 in self.edges:
            if p1 == position:
                neighbors.append(p2)
            elif p2 == position:
                neighbors.append(p1)
        return neighbors

    def build_from_positions(
        self, positions: List[Position], edge_threshold: float = 0.30
    ) -> None:
        """Build the map from a list of positions, creating edges between nearby nodes.

        Nodes within edge_threshold distance of each other will be connected.

        Args:
            positions: List of (x, z) positions to add as nodes.
            edge_threshold: Maximum distance for creating an edge between nodes.
        """
        # Add all positions as nodes
        for pos in positions:
            self.add_node(pos)

        # Create edges between nearby nodes
        for i, p1 in enumerate(positions):
            for p2 in positions[i + 1 :]:
                if euclidean_distance(p1, p2) <= edge_threshold:
                    self.add_edge(p1, p2)

    def find_path(self, start: Position, goal: Position) -> Optional[List[Position]]:
        """Find the shortest path between two nodes using A* algorithm.

        Args:
            start: Starting position.
            goal: Goal position.

        Returns:
            List of positions forming the path from start to goal, or None if
            no path exists.
        """
        # Check if both nodes exist
        if start not in self.nodes or goal not in self.nodes:
            return None

        # Special case: start equals goal
        if start == goal:
            return [start]

        # A* algorithm
        # g_score: cost from start to current node
        g_score: Dict[Position, float] = {start: 0.0}
        # came_from: parent node for reconstructing path
        came_from: Dict[Position, Position] = {}

        # Priority queue: (f_score, counter, position)
        # counter is used to break ties and ensure stable ordering
        counter = 0
        open_set: List[Tuple[float, int, Position]] = [
            (euclidean_distance(start, goal), counter, start)
        ]
        open_set_hash: Set[Position] = {start}

        while open_set:
            _, _, current = heapq.heappop(open_set)
            open_set_hash.discard(current)

            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return list(reversed(path))

            for neighbor in self.get_neighbors(current):
                # g_score for neighbor is g_score of current + distance
                tentative_g = g_score[current] + euclidean_distance(current, neighbor)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # This path to neighbor is better
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + euclidean_distance(neighbor, goal)

                    if neighbor not in open_set_hash:
                        counter += 1
                        heapq.heappush(open_set, (f_score, counter, neighbor))
                        open_set_hash.add(neighbor)

        # No path found
        return None

    def get_nearest_node(self, position: Position) -> Optional[Position]:
        """Find the nearest node to a given position.

        Args:
            position: The query position.

        Returns:
            The position of the nearest node, or None if the map is empty.
        """
        if not self.nodes:
            return None

        nearest = None
        min_dist = float("inf")

        for node_pos in self.nodes:
            dist = euclidean_distance(position, node_pos)
            if dist < min_dist:
                min_dist = dist
                nearest = node_pos

        return nearest

    def get_nodes_with_object(self, object_type: str) -> List[Position]:
        """Find all nodes where a specific object type has been seen.

        Args:
            object_type: The type of object to search for.

        Returns:
            List of positions where the object has been observed.
        """
        result = []
        for pos, node in self.nodes.items():
            if object_type in node.objects_seen:
                result.append(pos)
        return result
