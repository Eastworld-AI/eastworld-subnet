# The MIT License (MIT)
# Copyright © 2025 Eastworld AI

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import collections
import heapq

import bittensor as bt
import numpy as np

ANONYMOUS_NODE_PREFIX = "node_"


def heuristic(a, b):
    """Heuristic function: Manhattan distance"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class OccupancyGridMap:
    def __init__(self, width: int = 400, height: int = 400, resolution: float = 5.0):
        """Initialize an occupancy grid map"""
        self.width = width
        self.height = height
        self.resolution = resolution

        self.grid = np.zeros((height, width))
        self.base_offset_x = 0
        self.base_offset_y = 0

        # Use log-odds representation to store occupancy probability
        self.log_odds_occupied = 0.9  # update value when occupied
        self.log_odds_free = -0.6  # update value when free
        self.log_odds_threshold = 0.5  # threshold for occupied probability

        self.nav_nodes = {}
        self.nav_edges = collections.defaultdict(dict)

    def reset(self):
        """Reset the grid map to all unknown"""
        self.grid.fill(0)

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """Convert world coordinates to grid coordinates"""
        grid_x = int(x // self.resolution + self.width // 2 + self.base_offset_x)
        grid_y = int(y // self.resolution + self.height // 2 + self.base_offset_y)
        return min(max(0, grid_x), self.width - 1), min(max(0, grid_y), self.height - 1)

    def grid_to_world(self, grid_x: int, grid_y: int) -> tuple[float, float]:
        """Convert grid coordinates to world coordinates"""
        x = (grid_x - self.width // 2 - self.base_offset_x) * self.resolution
        y = (grid_y - self.height // 2 - self.base_offset_y) * self.resolution
        return x, y

    def update_cell(self, grid_x: int, grid_y: int, occupied: bool):
        """Update the occupancy probability of a cell"""
        if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
            if occupied:
                self.grid[grid_y, grid_x] += self.log_odds_occupied
            else:
                self.grid[grid_y, grid_x] += self.log_odds_free

    def is_occupied(self, grid_x: int, grid_y: int) -> bool | None:
        """Check if a cell is occupied"""
        if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
            return self.grid[grid_y, grid_x] > self.log_odds_threshold
        return None

    def expand_map(self, new_width=None, new_height=None):
        """Expand the map size by a default factor of 1.4"""
        # If new size is not specified, expand by 40%
        if new_width is None:
            new_width = int(self.width * 1.4)
        if new_height is None:
            new_height = int(self.height * 1.4)

        # Ensure new size is greater than old size
        new_width = max(new_width, self.width)
        new_height = max(new_height, self.height)

        # Calculate offsets
        x_offset = (new_width - self.width) // 2
        y_offset = (new_height - self.height) // 2

        # Create a new grid with the new size
        new_grid = np.zeros((new_height, new_width))

        # Copy the old map data to the new map
        new_grid[
            y_offset : y_offset + self.height, x_offset : x_offset + self.width
        ] = self.grid

        # Update map attributes
        self.grid = new_grid

        # Update base offsets to maintain world coordinate consistency
        self.base_offset_x += x_offset
        self.base_offset_y += y_offset

        self.width = new_width
        self.height = new_height

        return x_offset, y_offset  # Return the offset for the original map

    def justify_map(self, factor: float = 1.4):
        """
        Readjust the map size to center the current content and leave space according to the factor
        """
        # Find the range of areas with content on the map
        occupied_cells = []
        threshold = 0.1  # Use a threshold to determine which cells are considered "with content"

        for y in range(self.height):
            for x in range(self.width):
                # Check if the cell has content (not in unknown state)
                if abs(self.grid[y, x]) > threshold:
                    occupied_cells.append((x, y))

        # If there are no cells with content, keep the map unchanged
        if not occupied_cells:
            return

        # Determine the boundaries of the area with content
        min_x = min(x for x, _ in occupied_cells)
        max_x = max(x for x, _ in occupied_cells)
        min_y = min(y for _, y in occupied_cells)
        max_y = max(y for _, y in occupied_cells)

        # Calculate the width and height of the area with content
        content_width = max_x - min_x + 1
        content_height = max_y - min_y + 1

        # Apply the factor to calculate the width and height of the new map
        new_width = int(content_width * factor)
        new_height = int(content_height * factor)

        # Ensure the new map size is not smaller than the content area
        new_width = max(new_width, content_width, new_height, content_height)
        new_height = new_width

        # Create a new map
        new_grid = np.zeros((new_height, new_width))

        # Calculate the offset of the content in the new map to center it
        x_offset = (new_width - content_width) // 2
        y_offset = (new_height - content_height) // 2

        # Copy the area with content to the new map
        for old_y in range(min_y, max_y + 1):
            for old_x in range(min_x, max_x + 1):
                if 0 <= old_y < self.height and 0 <= old_x < self.width:
                    new_y = y_offset + (old_y - min_y)
                    new_x = x_offset + (old_x - min_x)
                    if 0 <= new_y < new_height and 0 <= new_x < new_width:
                        new_grid[new_y, new_x] = self.grid[old_y, old_x]

        # Save the dimensions and offsets of the old map for subsequent calculations
        old_width = self.width
        old_height = self.height
        old_offset_x = self.base_offset_x
        old_offset_y = self.base_offset_y

        # Update map attributes
        self.grid = new_grid

        # Calculate the position of the origin in the new map
        old_origin_x = old_width // 2 + old_offset_x
        old_origin_y = old_height // 2 + old_offset_y
        new_origin_x = x_offset + (old_origin_x - min_x)
        new_origin_y = y_offset + (old_origin_y - min_y)

        # Update offsets:
        self.base_offset_x = new_origin_x - (new_width // 2)
        self.base_offset_y = new_origin_y - (new_height // 2)

        self.width = new_width
        self.height = new_height

    def _add_nav_edge(self, node_id1: str, node_id2: str, cost: float = 1.0):
        if node_id1 not in self.nav_nodes or node_id2 not in self.nav_nodes:
            raise ValueError("One or both nodes do not exist")
        if node_id1 == node_id2:
            raise ValueError("Cannot connect a node to itself")

        self.nav_edges[node_id1][node_id2] = cost
        self.nav_edges[node_id2][node_id1] = cost

    def update_nav_topo(
        self,
        pose_index: int,
        x: float,
        y: float,
        node_id: str = None,
        node_desc: str = None,
        allow_isolated: bool = False,
    ):
        # Find nodes within a certain euclidean distance from the given point
        e_dist_threshold = 100.0
        node_candidates = []

        # Find nodes within euclidean distance threshold
        for nid, (_, nx, ny, _) in self.nav_nodes.items():
            # Directly use world coordinates to calculate Euclidean distance
            e_dist = ((x - nx) ** 2 + (y - ny) ** 2) ** 0.5
            if e_dist < e_dist_threshold:
                node_candidates.append((nid, nx, ny, e_dist))

        # Use dstar lite to find the nearest node by path distance
        dstar_max_path_length = 200
        nearest_node = None
        nearest_step = float("inf")

        # If there are candidate nodes, use D* Lite to find the nearest one by path
        for nid, nx, ny, _ in node_candidates:
            start_grid = self.world_to_grid(x, y)
            goal_grid = self.world_to_grid(nx, ny)

            # Find path using D* Lite
            path = self._dstar_lite_path(
                start_grid, goal_grid, max_path_length=dstar_max_path_length
            )

            # Calculate path distance
            if path:
                path_step = len(path)
                if path_step < nearest_step:
                    nearest_step = path_step
                    nearest_node = nid

        path_step_threshold = 15.0
        # If node_id is specified, create a new node anyway
        if node_id == nearest_node:
            bt.logging.warning(
                f"Duplicate node id {node_id}. No new navigation node added"
            )
        elif node_id is not None:
            if nearest_node is not None:
                self.nav_nodes[node_id] = (pose_index, x, y, node_desc)
                self._add_nav_edge(node_id, nearest_node, nearest_step)
                bt.logging.debug(
                    f"Added navigation node {node_id} with edge to {nearest_node}"
                )
            elif allow_isolated:
                self.nav_nodes[node_id] = (pose_index, x, y, node_desc)
                bt.logging.debug(f"Added isolated navigation node {node_id}")
        # If no candidates, add an isolated node
        elif not node_candidates and allow_isolated:
            node_id = f"{ANONYMOUS_NODE_PREFIX}{len(self.nav_nodes)}_{pose_index}"
            self.nav_nodes[node_id] = (pose_index, x, y, node_desc)
            bt.logging.debug(
                f"Added isolated navigation node {node_id} (no nearby nodes)"
            )
            return node_id
        # Else if nearest step is greater than threshold, add a new node
        elif nearest_node is not None and nearest_step > path_step_threshold:
            node_id = f"{ANONYMOUS_NODE_PREFIX}{len(self.nav_nodes)}_{pose_index}"
            self.nav_nodes[node_id] = (pose_index, x, y, node_desc)
            self._add_nav_edge(node_id, nearest_node, nearest_step)
            bt.logging.debug(
                f"Added anonymous navigation node {node_id} with edge to {nearest_node}"
            )
        else:
            bt.logging.trace(
                "No new navigation node added, empty candidates or too far from existing nodes"
            )

        return node_id

    def _find_frontier_cells(self, min_frontier_size=5) -> list[list[tuple[int, int]]]:
        """
        Find frontier cells in the map (boundary between known and unknown space)
        """
        # Values below this threshold are considered free space
        free_threshold = -0.1
        # Values close to zero are considered unknown space
        unknown_threshold = 0.1

        # Find all free space cells
        free_cells = []
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] < free_threshold:
                    free_cells.append((x, y))

        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        # Find all frontier cells (free space cells next to unknown space cells)
        frontier_cells = []
        for x, y in free_cells:
            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < self.width
                    and 0 <= ny < self.height
                    and -unknown_threshold < self.grid[ny, nx] < unknown_threshold
                ):
                    frontier_cells.append((x, y))
                    break

        # Cluster frontier cells to form continuous frontier regions
        frontiers = self._cluster_frontier_cells(frontier_cells)

        # Filter out frontier regions that are too small
        frontiers = [f for f in frontiers if len(f) >= min_frontier_size]

        return frontiers

    def _cluster_frontier_cells(self, frontier_cells) -> list[list[tuple[int, int]]]:
        """
        Cluster frontier cells to form continuous frontier regions
        """
        if not frontier_cells:
            return []

        # Mark frontier cells as processed
        processed = set()
        frontiers = []

        # Perform breadth-first search for each unprocessed frontier cell
        for cell in frontier_cells:
            if cell in processed:
                continue

            # Start a new frontier region
            frontier = []
            queue = [cell]
            processed.add(cell)

            while queue:
                current = queue.pop(0)
                frontier.append(current)

                # Check four-connected neighbors
                x, y = current
                neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

                for neighbor in neighbors:
                    if neighbor in frontier_cells and neighbor not in processed:
                        queue.append(neighbor)
                        processed.add(neighbor)

            frontiers.append(frontier)

        return frontiers

    def get_nearest_exploration_target(
        self, current_x, current_y, min_distance=5
    ) -> tuple[float, float] | None:
        """Get the best exploration target"""
        current_x, current_y = self.world_to_grid(current_x, current_y)

        frontiers = self._find_frontier_cells()
        if not frontiers:
            return None

        # Calculate the center point of each frontier region
        centers = []
        for frontier in frontiers:
            center_x = sum(x for x, _ in frontier) / len(frontier)
            center_y = sum(y for _, y in frontier) / len(frontier)
            centers.append((center_x, center_y))

        # Calculate the distance from the current position to each frontier center
        distances = []
        for center_x, center_y in centers:
            dist = ((center_x - current_x) ** 2 + (center_y - current_y) ** 2) ** 0.5
            if dist >= min_distance:
                distances.append(dist)

        if not distances:
            return None

        # # Select the nearest frontier center
        min_index = distances.index(min(distances))
        target = centers[min_index]

        return self.grid_to_world(target[0], target[1])

    def get_largest_exploration_target(
        self, current_x, current_y, min_distance=5
    ) -> tuple[float, float] | None:
        """Get the best exploration target"""
        current_x, current_y = self.world_to_grid(current_x, current_y)

        frontiers = self._find_frontier_cells()
        if not frontiers:
            return None

        # Calculate the center point of each frontier region
        centers = []
        for frontier in frontiers:
            center_x = sum(x for x, _ in frontier) / len(frontier)
            center_y = sum(y for _, y in frontier) / len(frontier)
            centers.append((center_x, center_y))

        sizes = []
        for frontier in frontiers:
            sizes.append(len(frontier))

        max_index = sizes.index(max(sizes))
        target = centers[max_index]

        return self.grid_to_world(target[0], target[1])

    def _get_neighbors(self, node, initialize_new=False, g=None, rhs=None):
        """
        Get the neighbors of a node, a generic version that can be used by different path planning algorithms

        Args:
            node: Current node coordinates (x, y)
            initialize_new: Whether to initialize newly discovered nodes (only applicable to D* Lite)
            g: Optional g-value dictionary (only applicable to D* Lite)
            rhs: Optional rhs-value dictionary (only applicable to D* Lite)

        Returns:
            Iterable of (neighbor, cost) pairs
        """
        # Define possible movement directions (8 directions)
        directions = [
            (0, 1),
            (1, 0),
            (0, -1),
            (-1, 0),  # Up, right, down, left
            (1, 1),
            (1, -1),
            (-1, -1),
            (-1, 1),  # Diagonals
        ]

        for dx, dy in directions:
            neighbor = (node[0] + dx, node[1] + dy)

            # Check boundaries
            if (
                neighbor[0] < 0
                or neighbor[0] >= self.width
                or neighbor[1] < 0
                or neighbor[1] >= self.height
            ):
                continue

            # Check if it's an obstacle
            if self.is_occupied(neighbor[0], neighbor[1]):
                continue

            # For diagonal movement, check if the adjacent two cells have obstacles
            if abs(dx) == 1 and abs(dy) == 1:
                if self.is_occupied(node[0] + dx, node[1]) or self.is_occupied(
                    node[0], node[1] + dy
                ):
                    continue

            # Calculate movement cost
            cost = 1.414 if abs(dx) + abs(dy) == 2 else 1

            # Initialize newly discovered nodes as needed (D* Lite specific feature)
            if (
                initialize_new
                and g is not None
                and rhs is not None
                and neighbor not in g
            ):
                g[neighbor] = float("inf")
                rhs[neighbor] = float("inf")

            yield neighbor, cost

    def _astar_path(self, start, goal, max_iterations=1000, max_path_length=1000):
        """
        Use A* algorithm to find a path from start to goal on the map

        Args:
            start: Start coordinates (x, y)
            goal: Goal coordinates (x, y)
            max_iterations: Maximum number of iterations to prevent infinite loops
            max_path_length: Maximum path length limit

        Returns:
            path: List of path points, or empty list if no path is found
        """
        # Input validation
        if not (0 <= start[0] < self.width and 0 <= start[1] < self.height):
            bt.logging.error(f"Invalid start position: {start}")
            return []
        if not (0 <= goal[0] < self.width and 0 <= goal[1] < self.height):
            bt.logging.error(f"Invalid goal position: {goal}")
            return []

        # Check if start and goal are obstacles
        if self.is_occupied(start[0], start[1]):
            bt.logging.error(f"Start position is occupied: {start}")
            return []
        if self.is_occupied(goal[0], goal[1]):
            bt.logging.error(f"Goal position is occupied: {goal}")
            return []

        # If start and goal are the same, return directly
        if start == goal:
            bt.logging.info("Start and goal positions are the same")
            return [start]

        # Initialize open list and closed list
        open_set = []
        closed_set = (
            set()
        )  # Add closed set to optimize checking if a node has been visited
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        open_set_hash = {start}

        # Unified heuristic function to ensure consistency
        def improved_heuristic(a, b):
            # Use more precise diagonal distance
            dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
            return 1.414 * min(dx, dy) + abs(dx - dy)

        visited_nodes = 0  # Track number of visited nodes for performance analysis
        iterations = 0

        bt.logging.debug(f"A* search started from {start} to {goal}")

        while open_set and iterations < max_iterations:
            iterations += 1

            if iterations % 1000 == 0:
                bt.logging.debug(
                    f"A* search: iteration {iterations}, visited {visited_nodes} nodes"
                )

            # Get the node with the smallest f_score
            current = heapq.heappop(open_set)[1]
            open_set_hash.remove(current)
            closed_set.add(current)
            visited_nodes += 1

            # If the goal is reached, construct the path and return
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()

                # Validate path length
                if len(path) > max_path_length:
                    bt.logging.warning(
                        f"A* found path exceeds maximum length of {max_path_length}"
                    )
                    return []

                bt.logging.info(
                    f"A* found path of length {len(path)} in {iterations} iterations"
                )
                return path

            # Check all adjacent nodes
            for neighbor, move_cost in self._get_neighbors(current):
                # Skip if the node is already in the closed list
                if neighbor in closed_set:
                    continue

                tentative_g_score = g_score[current] + move_cost

                # If a better path is found or it's a new node
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # Update path and scores
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + improved_heuristic(
                        neighbor, goal
                    )

                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)

        # If no path can be found, return an empty list
        bt.logging.warning(
            f"A* search failed to find a path after {iterations} iterations"
        )
        return []

    def _dstar_lite_path(self, start, goal, max_iterations=10000, max_path_length=1000):
        """
        Use D* Lite algorithm to find a path from start to goal on the map

        Args:
            start: Start coordinates (x, y)
            goal: Goal coordinates (x, y)
            max_iterations: Maximum number of iterations to prevent infinite loops
            max_path_length: Maximum path length limit

        Returns:
            path: List of path points, or empty list if no path is found
        """
        # Input validation
        if not (0 <= start[0] < self.width and 0 <= start[1] < self.height):
            bt.logging.error(f"Invalid start position: {start}")
            return []
        if not (0 <= goal[0] < self.width and 0 <= goal[1] < self.height):
            bt.logging.error(f"Invalid goal position: {goal}")
            return []

        # Check if start and goal are obstacles
        if self.is_occupied(start[0], start[1]):
            bt.logging.error(f"Start position is occupied: {start}")
            return []
        if self.is_occupied(goal[0], goal[1]):
            bt.logging.error(f"Goal position is occupied: {goal}")
            return []

        # If start and goal are the same, return directly
        if start == goal:
            bt.logging.info("Start and goal positions are the same")
            return [start]

        # Initialize data structures (lazy initialization)
        g = {}  # Actual cost from start to node
        rhs = {}  # Best estimated cost from start to node
        queue = []  # Priority queue
        in_queue = set()  # Track if a node is in the queue
        visited_nodes = set()  # Track visited nodes

        # Initialize start and goal nodes
        g[start] = float("inf")
        rhs[start] = float("inf")
        g[goal] = float("inf")
        rhs[goal] = 0  # Goal is the starting point of the search

        # Unified heuristic function to ensure consistency
        def cost(a, b):
            dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
            return 1.414 * min(dx, dy) + abs(dx - dy)

        # Calculate node's key value (priority)
        def calculate_key(node):
            k1 = min(g.get(node, float("inf")), rhs.get(node, float("inf"))) + cost(
                node, start
            )
            k2 = min(g.get(node, float("inf")), rhs.get(node, float("inf")))
            return (k1, k2)

        # Queue operations with lazy deletion strategy
        def insert(node):
            key = calculate_key(node)
            if node in in_queue:
                in_queue.remove(node)
            in_queue.add(node)
            heapq.heappush(queue, (key, node))

        # Get node from the queue
        def pop_node():
            while queue:
                key, node = heapq.heappop(queue)
                if node in in_queue:
                    in_queue.remove(node)
                    return key, node
            return None, None

        # Update node status
        def update_node(node):
            if g.get(node, float("inf")) != rhs.get(node, float("inf")):
                insert(node)

        # Get node's neighbors (using generic method)
        def get_valid_neighbors(node):
            return self._get_neighbors(node, initialize_new=True, g=g, rhs=rhs)

        # Calculate node's rhs value
        def compute_shortest_path():
            # Initialize goal node to the queue
            insert(goal)
            bt.logging.trace(
                f"D* Lite: Starting path computation from {goal} to {start}"
            )

            iterations = 0
            path_found = False

            while queue and iterations < max_iterations:
                iterations += 1

                if iterations % 1000 == 0:
                    bt.logging.trace(
                        f"D* Lite: Iteration {iterations}, queue size: {len(queue)}"
                    )

                # Get node with highest priority
                k_old, u = pop_node()

                if u is None:
                    break

                visited_nodes.add(u)

                # Check if we've reached the start or no path exists
                if u == start:
                    path_found = True
                    break

                if rhs.get(start, float("inf")) < float("inf"):
                    path_found = True
                    break

                k_new = calculate_key(u)

                # Check if priority has changed
                if k_old < k_new:
                    insert(u)
                elif g.get(u, float("inf")) > rhs.get(u, float("inf")):
                    g[u] = rhs[u]
                    # Update all affected nodes
                    for s, move_cost in get_valid_neighbors(u):
                        if s != goal:
                            new_rhs = min(rhs.get(s, float("inf")), g[u] + move_cost)
                            if new_rhs < rhs.get(s, float("inf")):
                                rhs[s] = new_rhs
                                update_node(s)
                else:
                    g_old = g.get(u, float("inf"))
                    g[u] = float("inf")

                    # Update the node and its predecessors
                    affected_nodes = [u]
                    for s, _ in get_valid_neighbors(u):
                        if s != goal and rhs.get(s, float("inf")) == g_old + _:
                            affected_nodes.append(s)

                    for s in affected_nodes:
                        if s != goal:
                            # Recalculate rhs value
                            min_rhs = float("inf")
                            for s_prime, move_cost in get_valid_neighbors(s):
                                min_rhs = min(
                                    min_rhs, g.get(s_prime, float("inf")) + move_cost
                                )
                            rhs[s] = min_rhs
                        update_node(s)

            if iterations >= max_iterations:
                bt.logging.warning(
                    f"D* Lite search reached max iterations ({max_iterations})"
                )
                return False

            bt.logging.trace(
                f"D* Lite: Visited {len(visited_nodes)} nodes in {iterations} iterations"
            )
            return path_found

        # Execute path planning
        if not compute_shortest_path():
            bt.logging.trace("D* Lite failed to find a path")
            return []

        # Check if a valid path was found
        if rhs.get(start, float("inf")) == float("inf"):
            bt.logging.error("D* Lite could not reach the start node from goal")
            return []

        # Build path
        path = [start]
        current = start
        path_length = 1
        visited_in_path = {start}  # Prevent cycles

        # Starting from the start point, greedily choose the next node
        while current != goal and path_length < max_path_length:
            # Find the best next step
            best_next = None
            min_cost = float("inf")

            for neighbor, move_cost in get_valid_neighbors(current):
                if neighbor in visited_in_path:
                    continue  # Skip already visited nodes to prevent cycles

                # Calculate the estimated total cost to the goal
                total_cost = move_cost + g.get(neighbor, float("inf"))

                if total_cost < min_cost:
                    min_cost = total_cost
                    best_next = neighbor

            # If next step can't be found, path planning fails
            if best_next is None:
                bt.logging.error(f"D* Lite path reconstruction failed at {current}")
                return []

            path.append(best_next)
            visited_in_path.add(best_next)
            current = best_next
            path_length += 1

            if path_length >= max_path_length:
                bt.logging.error(
                    f"D* Lite path exceeds maximum length of {max_path_length}"
                )
                return []

        bt.logging.trace(
            f"D* Lite found path of length {len(path)} from {start} to {goal}"
        )
        return path

    def node_navigation(
        self, node_start: str, node_end: str
    ) -> list[tuple[float, float]]:
        """Use A* to find the edges between two navigation nodes"""
        if node_start not in self.nav_nodes or node_end not in self.nav_nodes:
            raise ValueError("One or both nodes do not exist")

        # If start and end nodes are the same, directly return the node coordinates
        if node_start == node_end:
            return [self.nav_nodes[node_start][1:3]]

        # Initialize open list and closed list
        open_set = []
        visited = set()  # Set of visited nodes
        heapq.heappush(open_set, (0, node_start))

        # Record the cost from start to current node and the predecessor nodes
        g_score = {node_start: 0}
        came_from = {}

        def heuristic_nav(node1, node2):
            # using Euclidean distance between nodes
            pid1, x1, y1, _ = self.nav_nodes[node1]
            pid2, x2, y2, _ = self.nav_nodes[node2]
            return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

        def heuristic_cost(node1, node2):
            # Use cost of the edge if it exists, otherwise use heuristic_nav
            if node2 in self.nav_edges.get(node1, {}):
                return self.nav_edges[node1][node2]
            return heuristic_nav(node1, node2)

        # A* search main loop
        while open_set:
            # Get the node with the lowest score
            current_f, current_node = heapq.heappop(open_set)

            # Skip if this node has been visited
            if current_node in visited:
                continue

            # Add current node to visited set
            visited.add(current_node)

            # If target node reached, reconstruct path and return
            if current_node == node_end:
                # Reconstruct path
                path_nodes = []
                node = current_node
                while node in came_from:
                    path_nodes.append(node)
                    node = came_from[node]
                path_nodes.append(node_start)
                path_nodes.reverse()

                return [self.nav_nodes[node][1:3] for node in path_nodes]

            # Check all neighbors of current node
            for neighbor in self.nav_edges[current_node]:
                # Skip if already visited
                if neighbor in visited:
                    continue

                # Calculate cost from start through current node to neighbor
                tentative_g_score = (
                    g_score[current_node] + self.nav_edges[current_node][neighbor]
                )

                # If found better path or first time visiting this neighbor
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # Update path and cost
                    came_from[neighbor] = current_node
                    g_score[neighbor] = tentative_g_score
                    # 使用heuristic_cost代替heuristic_nav作为默认的启发式函数
                    f_score = tentative_g_score + heuristic_cost(neighbor, node_end)

                    # Add neighbor to open list
                    heapq.heappush(open_set, (f_score, neighbor))

        # If search ends but no path found, return empty list
        bt.logging.warning(f"No navigation path found from {node_start} to {node_end}")
        return []

    def pose_navigation(
        self, start_x, start_y, end_x, end_y
    ) -> list[tuple[float, float]]:
        """
        Get the navigation path from the given start and end world coordinates
        """
        # Find the nearest navigation nodes as start and end
        start_node = self._find_nearest_nav_node(start_x, start_y)
        end_node = self._find_nearest_nav_node(end_x, end_y)

        if not start_node or not end_node:
            bt.logging.warning("Could not find suitable navigation nodes")
            return []

        # Get the navigation path from the given start and end world coordinates
        nav_path = self.node_navigation(start_node, end_node)

        # Add start and end point to the path
        if nav_path:
            if (start_x, start_y) != nav_path[0]:
                nav_path.insert(0, (start_x, start_y))
            if (end_x, end_y) != nav_path[-1]:
                nav_path.append((end_x, end_y))

        return nav_path

    def _find_nearest_nav_node(self, x, y) -> str:
        """
        Find the nearest navigation node to the given world coordinates
        """
        if not self.nav_nodes:
            return None

        nearest_node = None
        min_distance = float("inf")

        for node_id, (_, node_x, node_y, _) in self.nav_nodes.items():
            distance = ((node_x - x) ** 2 + (node_y - y) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                nearest_node = node_id

        return nearest_node

    def get_nav_nodes(
        self, x: float = None, y: float = None, range: float = 20.0
    ) -> list[str]:
        if x is None or y is None:
            return list(self.nav_nodes.keys())

        nodes_in_radius = []
        for node_id, (_, node_x, node_y, node_desc) in self.nav_nodes.items():
            distance = ((node_x - x) ** 2 + (node_y - y) ** 2) ** 0.5
            if distance <= range:
                nodes_in_radius.append(node_id)
        return nodes_in_radius
