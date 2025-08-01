import random
from typing import Dict, List, Tuple, Optional, Set
from tqdm import tqdm
from collections import deque
import heapq
import json
from puzzle_engine import CubicPuzzle

class EnhancedAdaptiveSearchEngine:
    """
    Enhanced search engine with multiple algorithms and pattern database support
    """
    
    def __init__(self, knowledge_base: Dict[str, int], pattern_dbs: Dict = None,
                 cube_size: int = 3, depth_limit: int = 20):
        """
        Initialize the enhanced search engine
        
        Args:
            knowledge_base: Precomputed heuristic lookup table
            pattern_dbs: Pattern databases (corners, edges, centers)
            cube_size: Size of the cube (n for n×n×n)
            depth_limit: Maximum search depth allowed
        """
        self.depth_ceiling = depth_limit
        self.current_threshold = depth_limit
        self.next_threshold = None
        self.heuristic_db = knowledge_base
        self.pattern_dbs = pattern_dbs or {}
        self.cube_size = cube_size
        self.solution_path = []
        self.visited_states = set()
        self.move_cache = {}
        self.states_explored = 0
        self.algorithm_used = "None"
        
    def solve_puzzle(self, initial_state: str) -> List[Tuple[str, int, int]]:
        """Enhanced solver with multiple algorithms"""
        self.states_explored = 0
        
        # Check if already solved
        puzzle = CubicPuzzle(configuration=initial_state)
        if puzzle.is_completion_achieved():
            self.algorithm_used = "Already Solved"
            return []
        
        # First try BFS for shallow solutions (very fast for scrambles < 5 moves)
        print("Trying BFS for shallow solution...")
        bfs_solution = self._breadth_first_search(initial_state, max_depth=5)
        if bfs_solution is not None:
            self.algorithm_used = "Breadth-First Search"
            return bfs_solution
            
        # Then try bidirectional search for medium complexity
        print("Trying bidirectional search...")
        bidirectional_solution = self._bidirectional_search(initial_state)
        if bidirectional_solution is not None:
            self.algorithm_used = "Bidirectional Search"
            return bidirectional_solution
            
        # Fall back to IDA* for complex cases
        print("Using IDA* with pattern databases...")
        self.algorithm_used = "IDA* with Pattern Databases"
        return self._ida_star_search(initial_state)
    
    def _breadth_first_search(self, initial_state: str, max_depth: int) -> Optional[List[Tuple[str, int, int]]]:
        """Quick BFS for shallow solutions"""
        queue = deque([(initial_state, [])])
        visited = {initial_state}
        
        while queue:
            state, path = queue.popleft()
            self.states_explored += 1
            
            if len(path) > max_depth:
                return None
                
            puzzle = CubicPuzzle(configuration=state)
            if puzzle.is_completion_achieved():
                return path
                
            for move in self._generate_all_moves(self.cube_size):
                new_state = self._get_next_state(state, move)
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, path + [move]))
        
        return None
    
    def _bidirectional_search(self, initial_state: str) -> Optional[List[Tuple[str, int, int]]]:
        """Bidirectional search meeting in the middle"""
        # Get solved state
        temp_puzzle = CubicPuzzle(dimension=self.cube_size)
        solved_state = temp_puzzle.export_state()
        
        if initial_state == solved_state:
            return []
            
        # Forward and backward frontiers
        forward_frontier = {initial_state: []}
        backward_frontier = {solved_state: []}
        forward_visited = {initial_state: []}
        backward_visited = {solved_state: []}
        
        max_depth = self.depth_ceiling // 2
        
        for depth in range(max_depth):
            # Expand smaller frontier first
            if len(forward_frontier) <= len(backward_frontier):
                # Expand forward frontier
                new_forward = {}
                for state, path in forward_frontier.items():
                    self.states_explored += 1
                    for move in self._generate_ordered_moves(self.cube_size):
                        new_state = self._get_next_state(state, move)
                        
                        if new_state in backward_visited:
                            # Found connection!
                            backward_path = backward_visited[new_state]
                            return path + [move] + self._reverse_path(backward_path)
                            
                        if new_state not in forward_visited:
                            new_path = path + [move]
                            forward_visited[new_state] = new_path
                            new_forward[new_state] = new_path
                            
                forward_frontier = new_forward
            else:
                # Expand backward frontier
                new_backward = {}
                for state, path in backward_frontier.items():
                    self.states_explored += 1
                    for move in self._generate_ordered_moves(self.cube_size):
                        new_state = self._get_next_state(state, move)
                        
                        if new_state in forward_visited:
                            # Found connection!
                            forward_path = forward_visited[new_state]
                            return forward_path + self._reverse_path(path + [move])
                            
                        if new_state not in backward_visited:
                            new_path = path + [move]
                            backward_visited[new_state] = new_path
                            new_backward[new_state] = new_path
                            
                backward_frontier = new_backward
            
            # Stop if frontiers become too large
            if len(forward_frontier) == 0 or len(backward_frontier) == 0:
                break
                
        return None
    
    def _ida_star_search(self, initial_state: str) -> List[Tuple[str, int, int]]:
        """Optimized IDA* implementation with pattern databases"""
        self.visited_states.clear()
        self.solution_path = []
        
        # Get initial heuristic
        initial_h = self._get_enhanced_heuristic_value(initial_state)
        self.current_threshold = initial_h
        
        while self.current_threshold < self.depth_ceiling:
            self.next_threshold = float('inf')
            self.visited_states.clear()
            
            found_solution = self._depth_limited_search_optimized(initial_state, 0, None)
            if found_solution:
                return self.solution_path
                
            if self.next_threshold == float('inf'):
                break  # No solution found within depth limit
                
            self.current_threshold = self.next_threshold
            
        return []  # No solution found
    
    def _depth_limited_search_optimized(self, state: str, g: int, prev_move: Optional[Tuple] = None) -> bool:
        """Optimized recursive search with pruning"""
        self.states_explored += 1
        
        # Check if solved
        puzzle = CubicPuzzle(configuration=state)
        if puzzle.is_completion_achieved():
            return True
            
        # Calculate f-score
        h = self._get_enhanced_heuristic_value(state)
        f = g + h
        
        if f > self.current_threshold:
            if f < self.next_threshold:
                self.next_threshold = f
            return False
            
        # Avoid cycles
        if state in self.visited_states:
            return False
        self.visited_states.add(state)
        
        # Generate moves with smart ordering
        moves = self._generate_ordered_moves(self.cube_size, prev_move)
        
        # Try each move
        for move in moves:
            new_state = self._get_next_state(state, move)
            self.solution_path.append(move)
            
            if self._depth_limited_search_optimized(new_state, g + 1, move):
                return True
                
            self.solution_path.pop()
            
        self.visited_states.remove(state)
        return False
    
    def _get_enhanced_heuristic_value(self, state: str) -> int:
        """Enhanced heuristic using pattern databases and multiple strategies"""
        # First check full state database
        if state in self.heuristic_db:
            return self.heuristic_db[state]
        
        # Then use pattern databases
        if self.pattern_dbs:
            pattern_h = ScalablePatternDatabaseBuilder.get_pattern_heuristic(
                state, self.cube_size, self.pattern_dbs
            )
            if pattern_h < 20:  # Valid pattern found
                return pattern_h
        
        # Fall back to multi-size heuristic
        return MultiSizeHeuristic.calculate_heuristic(state, self.cube_size)
    
    def _generate_ordered_moves(self, puzzle_size: int, prev_move: Optional[Tuple] = None) -> List[Tuple[str, int, int]]:
        """Generate moves with smart ordering to find solution faster"""
        moves = self._generate_all_moves(puzzle_size)
        
        # Avoid immediately undoing the previous move
        if prev_move:
            inverse_move = self._get_inverse_move(prev_move)
            moves = [m for m in moves if m != inverse_move]
        
        # Prioritize moves based on cube size and layer
        def move_priority(move):
            move_type, layer, _ = move
            # Outer layers affect more pieces
            if layer == 0 or layer == puzzle_size - 1:
                return 0
            # Middle layers for odd-sized cubes
            elif puzzle_size % 2 == 1 and layer == puzzle_size // 2:
                return 1
            else:
                return 2
        
        return sorted(moves, key=move_priority)
    
    def _get_inverse_move(self, move: Tuple[str, int, int]) -> Tuple[str, int, int]:
        """Get the inverse of a move"""
        move_type, layer, direction = move
        return (move_type, layer, 1 - direction)
    
    def _get_next_state(self, state: str, move: Tuple[str, int, int]) -> str:
        """Get next state with caching"""
        cache_key = (state, move)
        if cache_key in self.move_cache:
            return self.move_cache[cache_key]
            
        puzzle = CubicPuzzle(configuration=state)
        self._apply_move(puzzle, move)
        new_state = puzzle.export_state()
        
        # Limit cache size to prevent memory issues
        if len(self.move_cache) < 50000:
            self.move_cache[cache_key] = new_state
            
        return new_state
    
    def _reverse_path(self, path: List[Tuple]) -> List[Tuple]:
        """Reverse a path by inverting moves in reverse order"""
        return [self._get_inverse_move(move) for move in reversed(path)]
    
    def _generate_all_moves(self, puzzle_size: int) -> List[Tuple[str, int, int]]:
        """Generate all possible moves for given puzzle size"""
        return [
            (move_type, layer, direction)
            for move_type in ['horizontal', 'vertical', 'sideways']
            for direction in [0, 1]
            for layer in range(puzzle_size)
        ]
    
    def _apply_move(self, puzzle: CubicPuzzle, move: Tuple[str, int, int]) -> None:
        """Apply a move to the puzzle"""
        move_type, layer, direction = move
        
        if move_type == 'horizontal':
            puzzle.execute_horizontal_rotation(layer, direction)
        elif move_type == 'vertical':
            puzzle.execute_vertical_rotation(layer, direction)
        elif move_type == 'sideways':
            puzzle.execute_lateral_rotation(layer, direction)
    
    def get_states_explored(self) -> int:
        """Get number of states explored during search"""
        return self.states_explored
    
    def get_algorithm_used(self) -> str:
        """Get the algorithm that found the solution"""
        return self.algorithm_used


class ScalablePatternDatabaseBuilder:
    """Pattern database builder that works for any n×n×n cube"""
    
    @staticmethod
    def get_corner_indices(n: int) -> Dict[str, List[Tuple[int, int, int]]]:
        """Generate corner indices for n×n×n cube"""
        last = n - 1
        return {
            'UFL': [(0, 0, 0), (1, 0, 0), (2, 0, 0)],
            'UFR': [(0, 0, last), (2, 0, last), (3, 0, 0)],
            'UBL': [(0, last, 0), (1, 0, last), (4, 0, last)],
            'UBR': [(0, last, last), (3, 0, last), (4, 0, 0)],
            'DFL': [(5, 0, 0), (1, last, 0), (2, last, 0)],
            'DFR': [(5, 0, last), (2, last, last), (3, last, 0)],
            'DBL': [(5, last, 0), (1, last, last), (4, last, last)],
            'DBR': [(5, last, last), (3, last, last), (4, last, 0)]
        }
    
    @staticmethod
    def get_edge_indices(n: int) -> Dict[str, List[List[Tuple[int, int, int]]]]:
        """Generate edge indices for n×n×n cube"""
        last = n - 1
        edges = {}
        
        # Top edges
        edges['UF'] = [[(0, 0, i), (2, 0, i)] for i in range(1, last)]
        edges['UR'] = [[(0, i, last), (3, 0, i)] for i in range(1, last)]
        edges['UB'] = [[(0, last, i), (4, 0, last-i)] for i in range(1, last)]
        edges['UL'] = [[(0, i, 0), (1, 0, i)] for i in range(1, last)]
        
        # Middle edges
        edges['FL'] = [[(2, i, 0), (1, i, 0)] for i in range(1, last)]
        edges['FR'] = [[(2, i, last), (3, i, 0)] for i in range(1, last)]
        edges['BL'] = [[(4, i, last), (1, i, last)] for i in range(1, last)]
        edges['BR'] = [[(4, i, 0), (3, i, last)] for i in range(1, last)]
        
        # Bottom edges
        edges['DF'] = [[(5, 0, i), (2, last, i)] for i in range(1, last)]
        edges['DR'] = [[(5, i, last), (3, last, i)] for i in range(1, last)]
        edges['DB'] = [[(5, last, i), (4, last, last-i)] for i in range(1, last)]
        edges['DL'] = [[(5, i, 0), (1, last, i)] for i in range(1, last)]
        
        return edges
    
    @staticmethod
    def get_center_indices(n: int) -> Dict[int, List[Tuple[int, int]]]:
        """Generate center indices for n×n×n cube (only for odd n)"""
        if n % 2 == 0:
            return {}
        
        mid = n // 2
        return {
            0: [(mid, mid)],  # Top
            1: [(mid, mid)],  # Left
            2: [(mid, mid)],  # Front
            3: [(mid, mid)],  # Right
            4: [(mid, mid)],  # Back
            5: [(mid, mid)]   # Bottom
        }
    
    @staticmethod
    def construct_heuristic_database(
        target_state: str,
        move_set: List[Tuple[str, int, int]],
        exploration_depth: int = 20,
        existing_knowledge: Optional[Dict[str, int]] = None
    ) -> Dict[str, int]:
        """Build comprehensive heuristic database with optimizations"""
        if existing_knowledge is None:
            knowledge_db = {target_state: 0}
        else:
            knowledge_db = existing_knowledge.copy()
        
        # Use deque for BFS (more efficient than list)
        exploration_queue = deque([(target_state, 0)])
        
        # Pre-calculate total nodes for progress tracking
        branching_factor = len(move_set)
        total_nodes = sum(branching_factor ** d for d in range(1, min(exploration_depth + 1, 5)))
        
        with tqdm(total=total_nodes, desc='Building Knowledge Base') as progress_bar:
            while exploration_queue:
                current_state, current_depth = exploration_queue.popleft()
                
                if current_depth >= exploration_depth:
                    continue
                
                for move in move_set:
                    puzzle = CubicPuzzle(configuration=current_state)
                    
                    # Apply move
                    if move[0] == 'horizontal':
                        puzzle.execute_horizontal_rotation(move[1], move[2])
                    elif move[0] == 'vertical':
                        puzzle.execute_vertical_rotation(move[1], move[2])
                    elif move[0] == 'sideways':
                        puzzle.execute_lateral_rotation(move[1], move[2])
                    
                    new_state = puzzle.export_state()
                    new_distance = current_depth + 1
                    
                    # Only add if we found a shorter path
                    if new_state not in knowledge_db or knowledge_db[new_state] > new_distance:
                        knowledge_db[new_state] = new_distance
                        exploration_queue.append((new_state, new_distance))
                        
                    progress_bar.update(1)
        
        return knowledge_db
    
    @staticmethod
    def build_pattern_database(target_state: str, n: int, pattern_type: str = "corners", 
                             max_depth: int = 10) -> Dict[str, int]:
        """
        Build pattern databases for n×n×n cubes
        
        Args:
            target_state: The solved state configuration
            n: Size of the cube
            pattern_type: "corners", "edges", "centers", or "combined"
            max_depth: Maximum depth to explore
        """
        if pattern_type == "corners":
            return ScalablePatternDatabaseBuilder._build_corner_database(target_state, n, max_depth)
        elif pattern_type == "edges":
            return ScalablePatternDatabaseBuilder._build_edge_database(target_state, n, max_depth)
        elif pattern_type == "centers" and n % 2 == 1:
            return ScalablePatternDatabaseBuilder._build_center_database(target_state, n, max_depth)
        elif pattern_type == "combined":
            result = {}
            result["corners"] = ScalablePatternDatabaseBuilder._build_corner_database(target_state, n, max_depth)
            if n == 3:  # Only build full edge DB for 3×3×3
                result["edges"] = ScalablePatternDatabaseBuilder._build_edge_database(target_state, n, max_depth)
            if n % 2 == 1:
                result["centers"] = ScalablePatternDatabaseBuilder._build_center_database(target_state, n, max_depth)
            return result
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
    
    @staticmethod
    def _build_corner_database(target_state: str, n: int, max_depth: int) -> Dict[str, int]:
        """Build corner pattern database for n×n×n cube"""
        print(f"Building corner pattern database for {n}×{n}×{n} cube...")
        
        corner_indices = ScalablePatternDatabaseBuilder.get_corner_indices(n)
        target_pattern = ScalablePatternDatabaseBuilder._extract_corner_pattern(target_state, n, corner_indices)
        
        pattern_db = {target_pattern: 0}
        queue = deque([(target_state, target_pattern, 0)])
        
        # Limit database size for memory efficiency
        max_patterns = 100000 if n > 3 else 1000000
        
        with tqdm(total=max_patterns, desc='Corner patterns') as pbar:
            while queue and len(pattern_db) < max_patterns:
                state, pattern, depth = queue.popleft()
                
                if depth >= max_depth:
                    continue
                
                puzzle = CubicPuzzle(configuration=state)
                moves = ScalablePatternDatabaseBuilder._generate_all_moves(n)
                
                for move in moves:
                    new_puzzle = CubicPuzzle(configuration=state)
                    ScalablePatternDatabaseBuilder._apply_move(new_puzzle, move)
                    new_state = new_puzzle.export_state()
                    
                    new_pattern = ScalablePatternDatabaseBuilder._extract_corner_pattern(new_state, n, corner_indices)
                    
                    if new_pattern not in pattern_db or pattern_db[new_pattern] > depth + 1:
                        pattern_db[new_pattern] = depth + 1
                        queue.append((new_state, new_pattern, depth + 1))
                        pbar.update(1)
        
        print(f"Corner database built with {len(pattern_db)} patterns")
        return pattern_db
    
    @staticmethod
    def _build_edge_database(target_state: str, n: int, max_depth: int) -> Dict[str, int]:
        """Build edge pattern database for n×n×n cube"""
        print(f"Building edge pattern database for {n}×{n}×{n} cube...")
        
        edge_indices = ScalablePatternDatabaseBuilder.get_edge_indices(n)
        
        # For memory efficiency on larger cubes, only track first 4-6 edges
        max_edges = 6 if n > 3 else 12
        selected_edges = dict(list(edge_indices.items())[:max_edges])
        
        target_pattern = ScalablePatternDatabaseBuilder._extract_edge_pattern(target_state, n, selected_edges)
        
        pattern_db = {target_pattern: 0}
        queue = deque([(target_state, target_pattern, 0)])
        
        max_patterns = 50000 if n > 3 else 500000
        
        with tqdm(total=max_patterns, desc='Edge patterns') as pbar:
            while queue and len(pattern_db) < max_patterns:
                state, pattern, depth = queue.popleft()
                
                if depth >= max_depth:
                    continue
                
                puzzle = CubicPuzzle(configuration=state)
                moves = ScalablePatternDatabaseBuilder._generate_all_moves(n)
                
                for move in moves:
                    new_puzzle = CubicPuzzle(configuration=state)
                    ScalablePatternDatabaseBuilder._apply_move(new_puzzle, move)
                    new_state = new_puzzle.export_state()
                    
                    new_pattern = ScalablePatternDatabaseBuilder._extract_edge_pattern(new_state, n, selected_edges)
                    
                    if new_pattern not in pattern_db or pattern_db[new_pattern] > depth + 1:
                        pattern_db[new_pattern] = depth + 1
                        queue.append((new_state, new_pattern, depth + 1))
                        pbar.update(1)
        
        print(f"Edge database built with {len(pattern_db)} patterns")
        return pattern_db
    
    @staticmethod
    def _build_center_database(target_state: str, n: int, max_depth: int) -> Dict[str, int]:
        """Build center pattern database for odd n×n×n cubes"""
        print(f"Building center pattern database for {n}×{n}×{n} cube...")
        
        center_indices = ScalablePatternDatabaseBuilder.get_center_indices(n)
        target_pattern = ScalablePatternDatabaseBuilder._extract_center_pattern(target_state, n, center_indices)
        
        pattern_db = {target_pattern: 0}
        queue = deque([(target_state, target_pattern, 0)])
        
        with tqdm(desc='Center patterns') as pbar:
            while queue and len(pattern_db) < 10000:
                state, pattern, depth = queue.popleft()
                
                if depth >= max_depth:
                    continue
                
                puzzle = CubicPuzzle(configuration=state)
                moves = ScalablePatternDatabaseBuilder._generate_all_moves(n)
                
                for move in moves:
                    new_puzzle = CubicPuzzle(configuration=state)
                    ScalablePatternDatabaseBuilder._apply_move(new_puzzle, move)
                    new_state = new_puzzle.export_state()
                    
                    new_pattern = ScalablePatternDatabaseBuilder._extract_center_pattern(new_state, n, center_indices)
                    
                    if new_pattern not in pattern_db or pattern_db[new_pattern] > depth + 1:
                        pattern_db[new_pattern] = depth + 1
                        queue.append((new_state, new_pattern, depth + 1))
                        pbar.update(1)
        
        print(f"Center database built with {len(pattern_db)} patterns")
        return pattern_db
    
    @staticmethod
    def _extract_corner_pattern(state: str, n: int, corner_indices: Dict) -> str:
        """Extract corner pattern from state"""
        puzzle = CubicPuzzle(configuration=state)
        pattern_parts = []
        
        for corner_name, positions in corner_indices.items():
            corner_colors = []
            for face, row, col in positions:
                corner_colors.append(puzzle.matrix[face][row][col])
            
            corner_sig = ''.join(sorted(corner_colors))
            pattern_parts.append(corner_sig)
        
        return '|'.join(pattern_parts)
    
    @staticmethod
    def _extract_edge_pattern(state: str, n: int, edge_indices: Dict) -> str:
        """Extract edge pattern from state"""
        puzzle = CubicPuzzle(configuration=state)
        pattern_parts = []
        
        for edge_name, edge_pieces in edge_indices.items():
            edge_signature = []
            for piece_positions in edge_pieces:
                piece_colors = []
                for face, row, col in piece_positions:
                    piece_colors.append(puzzle.matrix[face][row][col])
                edge_signature.append(''.join(sorted(piece_colors)))
            
            pattern_parts.append('-'.join(edge_signature))
        
        return '|'.join(pattern_parts)
    
    @staticmethod
    def _extract_center_pattern(state: str, n: int, center_indices: Dict) -> str:
        """Extract center pattern from state"""
        puzzle = CubicPuzzle(configuration=state)
        pattern_parts = []
        
        for face, positions in center_indices.items():
            for row, col in positions:
                pattern_parts.append(puzzle.matrix[face][row][col])
        
        return ''.join(pattern_parts)
    
    @staticmethod
    def get_pattern_heuristic(state: str, n: int, pattern_dbs: Dict) -> int:
        """Calculate heuristic using available pattern databases"""
        heuristics = []
        
        if "corners" in pattern_dbs:
            corner_indices = ScalablePatternDatabaseBuilder.get_corner_indices(n)
            corner_pattern = ScalablePatternDatabaseBuilder._extract_corner_pattern(state, n, corner_indices)
            corner_dist = pattern_dbs["corners"].get(corner_pattern, 20)
            heuristics.append(corner_dist)
        
        if "edges" in pattern_dbs and n == 3:  # Only for 3×3×3
            edge_indices = ScalablePatternDatabaseBuilder.get_edge_indices(n)
            selected_edges = dict(list(edge_indices.items())[:6])
            edge_pattern = ScalablePatternDatabaseBuilder._extract_edge_pattern(state, n, selected_edges)
            edge_dist = pattern_dbs["edges"].get(edge_pattern, 20)
            heuristics.append(edge_dist)
        
        if "centers" in pattern_dbs and n % 2 == 1:
            center_indices = ScalablePatternDatabaseBuilder.get_center_indices(n)
            center_pattern = ScalablePatternDatabaseBuilder._extract_center_pattern(state, n, center_indices)
            center_dist = pattern_dbs["centers"].get(center_pattern, 10)
            heuristics.append(center_dist)
        
        return max(heuristics) if heuristics else 0
    
    @staticmethod
    def _generate_all_moves(n: int) -> List[Tuple[str, int, int]]:
        """Generate all possible moves for n×n×n cube"""
        return [
            (move_type, layer, direction)
            for move_type in ['horizontal', 'vertical', 'sideways']
            for direction in [0, 1]
            for layer in range(n)
        ]
    
    @staticmethod
    def _apply_move(puzzle: CubicPuzzle, move: Tuple[str, int, int]) -> None:
        """Apply a move to the puzzle"""
        move_type, layer, direction = move
        
        if move_type == 'horizontal':
            puzzle.execute_horizontal_rotation(layer, direction)
        elif move_type == 'vertical':
            puzzle.execute_vertical_rotation(layer, direction)
        elif move_type == 'sideways':
            puzzle.execute_lateral_rotation(layer, direction)


class MultiSizeHeuristic:
    """Heuristic calculator that adapts to cube size"""
    
    @staticmethod
    def calculate_heuristic(state: str, n: int, pattern_dbs: Dict = None) -> int:
        """
        Calculate heuristic based on cube size
        
        For larger cubes, uses different strategies:
        - 2×2×2: Corner pattern database
        - 3×3×3: Corner + edge pattern databases
        - 4×4×4+: Corner + center parity + reduction heuristic
        """
        if pattern_dbs:
            pattern_h = ScalablePatternDatabaseBuilder.get_pattern_heuristic(state, n, pattern_dbs)
            if pattern_h < 20:
                return pattern_h
        
        # Fall back to size-specific heuristics
        if n == 2:
            return MultiSizeHeuristic._heuristic_2x2(state)
        elif n == 3:
            return MultiSizeHeuristic._heuristic_3x3(state)
        else:
            return MultiSizeHeuristic._heuristic_nxn(state, n)
    
    @staticmethod
    def _heuristic_2x2(state: str) -> int:
        """Heuristic for 2×2×2 cube (only corners matter)"""
        puzzle = CubicPuzzle(configuration=state)
        solved = CubicPuzzle(dimension=2)
        
        misplaced = 0
        for face in range(6):
            for row in range(2):
                for col in range(2):
                    if puzzle.matrix[face][row][col] != solved.matrix[face][row][col]:
                        misplaced += 1
        
        # Each move affects 4 corners
        return misplaced // 4
    
    @staticmethod
    def _heuristic_3x3(state: str) -> int:
        """Standard heuristic for 3×3×3 cube"""
        puzzle = CubicPuzzle(configuration=state)
        solved = CubicPuzzle(dimension=3)
        
        misplaced = 0
        for face in range(6):
            for row in range(3):
                for col in range(3):
                    if puzzle.matrix[face][row][col] != solved.matrix[face][row][col]:
                        misplaced += 1
        
        return misplaced // 4
    
    @staticmethod
    def _heuristic_nxn(state: str, n: int) -> int:
        """
        Heuristic for larger n×n×n cubes
        Uses reduction method concepts
        """
        puzzle = CubicPuzzle(configuration=state)
        solved = CubicPuzzle(dimension=n)
        
        # Count misplaced pieces by type
        corner_misplaced = 0
        edge_misplaced = 0
        center_misplaced = 0
        
        for face in range(6):
            for row in range(n):
                for col in range(n):
                    if puzzle.matrix[face][row][col] != solved.matrix[face][row][col]:
                        # Classify piece type
                        is_corner = (row in [0, n-1]) and (col in [0, n-1])
                        is_edge = ((row in [0, n-1]) ^ (col in [0, n-1])) and not is_corner
                        is_center = not is_corner and not is_edge
                        
                        if is_corner:
                            corner_misplaced += 1
                        elif is_edge:
                            edge_misplaced += 1
                        else:
                            center_misplaced += 1
        
        # Weighted heuristic based on piece type
        corner_h = corner_misplaced // 4
        edge_h = edge_misplaced // (2 * (n - 2)) if n > 2 else 0
        center_h = center_misplaced // (4 * (n - 2)) if n > 2 else 0
        
        return max(corner_h, edge_h // 2, center_h // 3)