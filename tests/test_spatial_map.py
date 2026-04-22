"""Tests for spatial_map module."""

import pytest
from datetime import datetime, timedelta

from src.memory.spatial_map import (
    Position,
    MapNode,
    TopologicalMap,
    euclidean_distance,
)


class TestMapNode:
    """Tests for MapNode dataclass."""

    def test_creation(self):
        """Test creating a MapNode with default values."""
        position: Position = (1.5, 2.5)
        node = MapNode(position=position)

        assert node.position == position
        assert node.objects_seen == set()
        assert node.visited_count == 0
        assert node.last_visited is None

    def test_creation_with_values(self):
        """Test creating a MapNode with specified values."""
        position: Position = (0.0, 0.0)
        objects_seen = {"apple", "cup"}
        now = datetime.now()

        node = MapNode(
            position=position,
            objects_seen=objects_seen,
            visited_count=3,
            last_visited=now,
        )

        assert node.position == position
        assert node.objects_seen == objects_seen
        assert node.visited_count == 3
        assert node.last_visited == now

    def test_add_object(self):
        """Test adding an object to objects_seen."""
        node = MapNode(position=(0.0, 0.0))

        node.add_object("apple")
        assert "apple" in node.objects_seen

        node.add_object("cup")
        assert "cup" in node.objects_seen
        assert len(node.objects_seen) == 2

    def test_add_object_duplicate(self):
        """Test adding the same object twice doesn't duplicate."""
        node = MapNode(position=(0.0, 0.0))

        node.add_object("apple")
        node.add_object("apple")

        assert len(node.objects_seen) == 1
        assert "apple" in node.objects_seen

    def test_mark_visited(self):
        """Test marking a node as visited."""
        node = MapNode(position=(0.0, 0.0))

        before = datetime.now()
        node.mark_visited()
        after = datetime.now()

        assert node.visited_count == 1
        assert before <= node.last_visited <= after

        node.mark_visited()
        assert node.visited_count == 2


class TestEuclideanDistance:
    """Tests for euclidean_distance function."""

    def test_same_point(self):
        """Test distance between same point is zero."""
        p: Position = (1.0, 2.0)
        assert euclidean_distance(p, p) == 0.0

    def test_horizontal_distance(self):
        """Test horizontal distance."""
        p1: Position = (0.0, 0.0)
        p2: Position = (3.0, 0.0)
        assert euclidean_distance(p1, p2) == 3.0

    def test_vertical_distance(self):
        """Test vertical distance."""
        p1: Position = (0.0, 0.0)
        p2: Position = (0.0, 4.0)
        assert euclidean_distance(p1, p2) == 4.0

    def test_diagonal_distance(self):
        """Test diagonal distance (3-4-5 triangle)."""
        p1: Position = (0.0, 0.0)
        p2: Position = (3.0, 4.0)
        assert euclidean_distance(p1, p2) == 5.0

    def test_negative_coordinates(self):
        """Test with negative coordinates."""
        p1: Position = (-1.0, -1.0)
        p2: Position = (2.0, 3.0)
        # sqrt(3^2 + 4^2) = 5
        assert euclidean_distance(p1, p2) == 5.0


class TestTopologicalMap:
    """Tests for TopologicalMap class."""

    def test_empty_map(self):
        """Test creating an empty map."""
        topo_map = TopologicalMap()

        assert len(topo_map.nodes) == 0
        assert len(topo_map.edges) == 0

    def test_add_node(self):
        """Test adding a node to the map."""
        topo_map = TopologicalMap()
        position: Position = (1.0, 2.0)

        node = topo_map.add_node(position)

        assert len(topo_map.nodes) == 1
        assert position in topo_map.nodes
        assert topo_map.nodes[position] == node
        assert node.position == position

    def test_add_node_existing_position(self):
        """Test adding a node at an existing position returns existing node."""
        topo_map = TopologicalMap()
        position: Position = (1.0, 2.0)

        node1 = topo_map.add_node(position)
        node2 = topo_map.add_node(position)

        assert node1 is node2
        assert len(topo_map.nodes) == 1

    def test_add_edge(self):
        """Test adding an edge between nodes."""
        topo_map = TopologicalMap()
        p1: Position = (0.0, 0.0)
        p2: Position = (1.0, 0.0)

        topo_map.add_node(p1)
        topo_map.add_node(p2)
        topo_map.add_edge(p1, p2)

        assert topo_map.has_edge(p1, p2)
        assert topo_map.has_edge(p2, p1)  # Edges are bidirectional

    def test_has_edge_no_edge(self):
        """Test has_edge returns False when no edge exists."""
        topo_map = TopologicalMap()
        p1: Position = (0.0, 0.0)
        p2: Position = (1.0, 0.0)

        topo_map.add_node(p1)
        topo_map.add_node(p2)

        assert not topo_map.has_edge(p1, p2)

    def test_get_neighbors(self):
        """Test getting neighbors of a node."""
        topo_map = TopologicalMap()
        p1: Position = (0.0, 0.0)
        p2: Position = (1.0, 0.0)
        p3: Position = (0.0, 1.0)

        topo_map.add_node(p1)
        topo_map.add_node(p2)
        topo_map.add_node(p3)
        topo_map.add_edge(p1, p2)
        topo_map.add_edge(p1, p3)

        neighbors = topo_map.get_neighbors(p1)
        assert len(neighbors) == 2
        assert p2 in neighbors
        assert p3 in neighbors

    def test_get_neighbors_no_neighbors(self):
        """Test getting neighbors of isolated node."""
        topo_map = TopologicalMap()
        p: Position = (0.0, 0.0)
        topo_map.add_node(p)

        neighbors = topo_map.get_neighbors(p)
        assert neighbors == []

    def test_build_from_positions(self):
        """Test building map from positions with edge threshold."""
        topo_map = TopologicalMap()
        positions = [
            (0.0, 0.0),
            (0.2, 0.0),  # distance 0.2 from first
            (0.5, 0.0),  # distance 0.5 from first
        ]

        topo_map.build_from_positions(positions, edge_threshold=0.30)

        # All nodes should be added
        assert len(topo_map.nodes) == 3

        # First and second should be connected (distance 0.2 < 0.30)
        assert topo_map.has_edge((0.0, 0.0), (0.2, 0.0))

        # Second and third should be connected (distance 0.3 == 0.30)
        assert topo_map.has_edge((0.2, 0.0), (0.5, 0.0))

        # First and third should NOT be connected (distance 0.5 > 0.30)
        assert not topo_map.has_edge((0.0, 0.0), (0.5, 0.0))

    def test_find_path_simple(self):
        """Test finding a simple path between two nodes."""
        topo_map = TopologicalMap()
        p1: Position = (0.0, 0.0)
        p2: Position = (1.0, 0.0)
        p3: Position = (2.0, 0.0)

        topo_map.add_node(p1)
        topo_map.add_node(p2)
        topo_map.add_node(p3)
        topo_map.add_edge(p1, p2)
        topo_map.add_edge(p2, p3)

        path = topo_map.find_path(p1, p3)

        assert path is not None
        assert path == [p1, p2, p3]

    def test_find_path_no_path(self):
        """Test finding path when no path exists."""
        topo_map = TopologicalMap()
        p1: Position = (0.0, 0.0)
        p2: Position = (2.0, 0.0)

        topo_map.add_node(p1)
        topo_map.add_node(p2)

        path = topo_map.find_path(p1, p2)

        assert path is None

    def test_find_path_same_start_goal(self):
        """Test finding path when start equals goal."""
        topo_map = TopologicalMap()
        p: Position = (0.0, 0.0)
        topo_map.add_node(p)

        path = topo_map.find_path(p, p)

        assert path == [p]

    def test_find_path_complex(self):
        """Test finding path in a more complex graph."""
        topo_map = TopologicalMap()
        # Create a grid-like graph:
        #  A -- B -- C
        #  |    |
        #  D -- E
        a: Position = (0.0, 0.0)
        b: Position = (1.0, 0.0)
        c: Position = (2.0, 0.0)
        d: Position = (0.0, 1.0)
        e: Position = (1.0, 1.0)

        topo_map.add_node(a)
        topo_map.add_node(b)
        topo_map.add_node(c)
        topo_map.add_node(d)
        topo_map.add_node(e)

        topo_map.add_edge(a, b)
        topo_map.add_edge(b, c)
        topo_map.add_edge(a, d)
        topo_map.add_edge(b, e)
        topo_map.add_edge(d, e)

        path = topo_map.find_path(d, c)

        assert path is not None
        assert path[0] == d
        assert path[-1] == c
        # Verify path is valid (each step is an edge)
        for i in range(len(path) - 1):
            assert topo_map.has_edge(path[i], path[i + 1])

    def test_find_path_nonexistent_node(self):
        """Test finding path when start or goal doesn't exist."""
        topo_map = TopologicalMap()
        p: Position = (0.0, 0.0)
        topo_map.add_node(p)

        path = topo_map.find_path(p, (1.0, 1.0))
        assert path is None

        path = topo_map.find_path((1.0, 1.0), p)
        assert path is None

    def test_get_nearest_node(self):
        """Test finding nearest node to a position."""
        topo_map = TopologicalMap()
        topo_map.add_node((0.0, 0.0))
        topo_map.add_node((1.0, 0.0))
        topo_map.add_node((0.0, 1.0))

        # Query position closest to (1.0, 0.0)
        nearest = topo_map.get_nearest_node((1.1, 0.1))
        assert nearest == (1.0, 0.0)

    def test_get_nearest_node_empty_map(self):
        """Test get_nearest_node on empty map."""
        topo_map = TopologicalMap()
        nearest = topo_map.get_nearest_node((0.0, 0.0))
        assert nearest is None

    def test_get_nodes_with_object(self):
        """Test finding nodes that have seen a specific object."""
        topo_map = TopologicalMap()

        n1 = topo_map.add_node((0.0, 0.0))
        n2 = topo_map.add_node((1.0, 0.0))
        n3 = topo_map.add_node((2.0, 0.0))

        n1.add_object("apple")
        n2.add_object("banana")
        n3.add_object("apple")

        apple_nodes = topo_map.get_nodes_with_object("apple")
        assert len(apple_nodes) == 2
        assert (0.0, 0.0) in apple_nodes
        assert (2.0, 0.0) in apple_nodes

        banana_nodes = topo_map.get_nodes_with_object("banana")
        assert len(banana_nodes) == 1
        assert (1.0, 0.0) in banana_nodes

    def test_get_nodes_with_object_not_found(self):
        """Test finding nodes with an object that hasn't been seen."""
        topo_map = TopologicalMap()
        topo_map.add_node((0.0, 0.0))

        nodes = topo_map.get_nodes_with_object("nonexistent")
        assert nodes == []
