import json
import os
import sys
import time
from typing import Dict, Tuple
from puzzle_engine import CubicPuzzle
from search_algorithm import EnhancedAdaptiveSearchEngine, ScalablePatternDatabaseBuilder

# Configuration Parameters
EXPLORATION_DEPTH = 5  # Increased for better solutions
REBUILD_KNOWLEDGE = False  # Set True to force rebuild
REBUILD_PATTERNS = False  # Set True to force pattern DB rebuild
DEFAULT_CUBE_SIZE = 3  # Default cube dimension

def main(cube_size: int = DEFAULT_CUBE_SIZE):
    """Main execution pipeline for the puzzle solving system"""
    
    print(f"=== Advanced {cube_size}×{cube_size}×{cube_size} Cube Solver ===")
    print("Initializing puzzle engine...")
    
    # Initialize puzzle system
    puzzle = CubicPuzzle(dimension=cube_size)
    puzzle.display_configuration()
    print('=' * 50)
    
    # Load or build all knowledge bases
    knowledge_db, pattern_dbs = load_or_build_knowledge_bases(puzzle, cube_size)
    
    # Create challenge scenario
    print("Generating puzzle challenge...")
    scramble_moves = calculate_scramble_moves(cube_size)
    puzzle.randomize_configuration(min_operations=scramble_moves, max_operations=scramble_moves + 5)
    
    print(f"Puzzle scrambled with ~{scramble_moves} moves")
    puzzle.display_configuration()
    print('=' * 50)
    
    # Solve the puzzle
    print("Initiating solution process...")
    start_time = time.time()
    
    # Create enhanced search engine with pattern databases
    search_engine = EnhancedAdaptiveSearchEngine(
        knowledge_base=knowledge_db,
        pattern_dbs=pattern_dbs,
        cube_size=cube_size,
        depth_limit=calculate_depth_limit(cube_size)
    )
    
    solution_sequence = search_engine.solve_puzzle(puzzle.export_state())
    solve_time = time.time() - start_time
    
    # Display results
    print(f"\n{'='*50}")
    print(f"SOLUTION FOUND!")
    print(f"{'='*50}")
    print(f"Algorithm used: {search_engine.get_algorithm_used()}")
    print(f"Solution length: {len(solution_sequence)} moves")
    print(f"States explored: {search_engine.get_states_explored():,}")
    print(f"Solve time: {solve_time:.2f} seconds")
    print(f"Efficiency rating: {calculate_efficiency(len(solution_sequence), scramble_moves)}")
    
    if len(solution_sequence) <= 20:
        print(f"\nSolution moves: {format_move_sequence(solution_sequence)}")
    else:
        print(f"\nFirst 20 moves: {format_move_sequence(solution_sequence[:20])}...")
    
    # Apply solution and verify
    print("\nApplying solution moves...")
    apply_solution_moves(puzzle, solution_sequence)
    puzzle.display_configuration()
    
    print(f"\nPuzzle solved: {puzzle.is_completion_achieved()}")
    print(f"{'='*50}")


def load_or_build_knowledge_bases(puzzle: CubicPuzzle, cube_size: int) -> Tuple[Dict, Dict]:
    """Load or build all knowledge bases including pattern databases"""
    
    # File names based on cube size
    knowledge_file = f'knowledge_base_{cube_size}x{cube_size}x{cube_size}.json'
    pattern_file = f'pattern_db_{cube_size}x{cube_size}x{cube_size}.json'
    
    # Load or build regular knowledge base
    knowledge_db = load_or_build_knowledge_base(puzzle, cube_size, knowledge_file)
    
    # Load or build pattern databases
    pattern_dbs = load_or_build_pattern_databases(puzzle, cube_size, pattern_file)
    
    return knowledge_db, pattern_dbs


def load_or_build_knowledge_base(puzzle: CubicPuzzle, cube_size: int, filename: str) -> Dict:
    """Load existing knowledge base or build new one"""
    
    knowledge_db = None
    
    # Try to load existing knowledge base
    if os.path.exists(filename) and not REBUILD_KNOWLEDGE:
        print(f"Loading knowledge base from {filename}...")
        try:
            with open(filename, 'r') as file:
                knowledge_db = json.load(file)
            print(f"Loaded {len(knowledge_db):,} state mappings")
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            knowledge_db = None
    
    # Build new knowledge base if needed
    if knowledge_db is None or REBUILD_KNOWLEDGE:
        print("Building new knowledge base...")
        
        move_catalog = generate_move_catalog(cube_size)
        
        # Adjust exploration depth based on cube size
        exploration_depth = get_exploration_depth(cube_size)
        
        knowledge_db = ScalablePatternDatabaseBuilder.construct_heuristic_database(
            target_state=puzzle.export_state(),
            move_set=move_catalog,
            exploration_depth=exploration_depth,
            existing_knowledge=knowledge_db
        )
        
        # Save knowledge base
        print(f"Saving knowledge base to {filename}...")
        try:
            with open(filename, 'w', encoding='utf-8') as file:
                json.dump(knowledge_db, file, ensure_ascii=False)
            print(f"Saved {len(knowledge_db):,} state mappings")
        except Exception as e:
            print(f"Error saving knowledge base: {e}")
    
    return knowledge_db


def load_or_build_pattern_databases(puzzle: CubicPuzzle, cube_size: int, filename: str) -> Dict:
    """Load or build pattern databases"""
    
    pattern_dbs = {}
    
    # Try to load existing pattern databases
    if os.path.exists(filename) and not REBUILD_PATTERNS:
        print(f"Loading pattern databases from {filename}...")
        try:
            with open(filename, 'r') as file:
                pattern_dbs = json.load(file)
            print(f"Loaded pattern databases: {list(pattern_dbs.keys())}")
        except Exception as e:
            print(f"Error loading pattern databases: {e}")
            pattern_dbs = {}
    
    # Build new pattern databases if needed
    if not pattern_dbs or REBUILD_PATTERNS:
        print(f"Building pattern databases for {cube_size}×{cube_size}×{cube_size} cube...")
        
        # Choose pattern types based on cube size
        if cube_size == 2:
            pattern_type = "corners"  # 2×2×2 only has corners
            max_depth = 11  # Can solve optimally
        elif cube_size == 3:
            pattern_type = "combined"  # Full patterns for 3×3×3
            max_depth = 8
        else:
            pattern_type = "corners"  # Only corners for larger cubes
            max_depth = 6
        
        pattern_dbs = ScalablePatternDatabaseBuilder.build_pattern_database(
            target_state=puzzle.export_state(),
            n=cube_size,
            pattern_type=pattern_type,
            max_depth=max_depth
        )
        
        # Save pattern databases
        print(f"Saving pattern databases to {filename}...")
        try:
            with open(filename, 'w', encoding='utf-8') as file:
                json.dump(pattern_dbs, file, ensure_ascii=False)
            
            total_patterns = sum(len(db) for db in pattern_dbs.values() if isinstance(db, dict))
            print(f"Saved {total_patterns:,} total patterns")
        except Exception as e:
            print(f"Error saving pattern databases: {e}")
    
    return pattern_dbs


def generate_move_catalog(puzzle_size: int) -> list:
    """Generate comprehensive catalog of all possible moves"""
    return [
        (rotation_type, layer_idx, direction)
        for rotation_type in ['horizontal', 'vertical', 'sideways']
        for direction in [0, 1]
        for layer_idx in range(puzzle_size)
    ]


def apply_solution_moves(puzzle: CubicPuzzle, move_sequence: list) -> None:
    """Apply sequence of moves to solve the puzzle"""
    
    for move in move_sequence:
        move_type, layer, direction = move
        
        if move_type == 'horizontal':
            puzzle.execute_horizontal_rotation(layer, direction)
        elif move_type == 'vertical':
            puzzle.execute_vertical_rotation(layer, direction)
        elif move_type == 'sideways':
            puzzle.execute_lateral_rotation(layer, direction)
        else:
            print(f"Warning: Unknown move type '{move_type}' - skipping")


def get_exploration_depth(cube_size: int) -> int:
    """Get appropriate exploration depth based on cube size"""
    depth_map = {
        2: 11,  # 2×2×2 can be solved optimally
        3: EXPLORATION_DEPTH,  # Use configured depth
        4: 4,   # Limited depth for 4×4×4
        5: 3    # Very limited for 5×5×5+
    }
    return depth_map.get(cube_size, 3)


def calculate_scramble_moves(cube_size: int) -> int:
    """Calculate appropriate scramble depth based on cube size"""
    scramble_map = {
        2: 10,
        3: 15,
        4: 25,
        5: 35
    }
    return scramble_map.get(cube_size, 20)


def calculate_depth_limit(cube_size: int) -> int:
    """Calculate search depth limit based on cube size"""
    limit_map = {
        2: 14,   # 2×2×2 God's number
        3: 20,   # 3×3×3 God's number
        4: 35,   # Estimated for 4×4×4
        5: 50    # Estimated for 5×5×5
    }
    return limit_map.get(cube_size, 30)


def calculate_efficiency(solution_length: int, scramble_moves: int) -> str:
    """Calculate efficiency rating"""
    if solution_length <= scramble_moves:
        return "⭐⭐⭐⭐⭐ Optimal"
    elif solution_length <= scramble_moves * 1.2:
        return "⭐⭐⭐⭐ Excellent"
    elif solution_length <= scramble_moves * 1.5:
        return "⭐⭐⭐ Good"
    else:
        return "⭐⭐ Fair"


def format_move_sequence(moves: list) -> str:
    """Format move sequence for display"""
    formatted = []
    for move_type, layer, direction in moves:
        move_char = move_type[0].upper()
        dir_char = "'" if direction == 0 else ""
        formatted.append(f"{move_char}{layer}{dir_char}")
    return " ".join(formatted)


def interactive_mode(cube_size: int = DEFAULT_CUBE_SIZE):
    """Interactive mode for manual puzzle manipulation"""
    
    puzzle = CubicPuzzle(dimension=cube_size)
    
    print(f"=== Interactive {cube_size}×{cube_size}×{cube_size} Puzzle Mode ===")
    print("Commands:")
    print("  h <layer> <direction> - Horizontal rotation")
    print("  v <layer> <direction> - Vertical rotation") 
    print("  s <layer> <direction> - Sideways rotation")
    print("  show - Display puzzle")
    print("  reset - Reset to solved state")
    print("  scramble [moves] - Randomize puzzle")
    print("  solve - Run AI solver")
    print("  quit - Exit")
    print(f"\nLayers: 0-{cube_size-1}, Direction: 0=CCW/down, 1=CW/up")
    
    while True:
        print("\n" + "="*50)
        puzzle.display_configuration()
        print(f"Solved: {puzzle.is_completion_achieved()}")
        
        try:
            command = input("\nEnter command: ").strip().lower().split()
            
            if not command:
                continue
            
            if command[0] == 'quit':
                break
            elif command[0] == 'show':
                continue
            elif command[0] == 'reset':
                puzzle.restore_factory_settings()
                print("Puzzle reset to solved state")
            elif command[0] == 'scramble':
                moves = int(command[1]) if len(command) > 1 else calculate_scramble_moves(cube_size)
                puzzle.randomize_configuration(min_operations=moves, max_operations=moves)
                print(f"Puzzle scrambled with {moves} moves")
            elif command[0] == 'solve':
                print("Running AI solver...")
                # Load knowledge bases
                knowledge_db, pattern_dbs = load_or_build_knowledge_bases(puzzle, cube_size)
                
                # Create search engine and solve
                search_engine = EnhancedAdaptiveSearchEngine(
                    knowledge_base=knowledge_db,
                    pattern_dbs=pattern_dbs,
                    cube_size=cube_size
                )
                
                start_time = time.time()
                solution = search_engine.solve_puzzle(puzzle.export_state())
                solve_time = time.time() - start_time
                
                print(f"Solution found: {len(solution)} moves in {solve_time:.2f}s")
                print(f"Moves: {format_move_sequence(solution[:20])}{'...' if len(solution) > 20 else ''}")
                
                if input("Apply solution? (y/n): ").lower() == 'y':
                    apply_solution_moves(puzzle, solution)
                    
            elif command[0] in ['h', 'v', 's'] and len(command) == 3:
                layer = int(command[1])
                direction = int(command[2])
                
                if command[0] == 'h':
                    puzzle.execute_horizontal_rotation(layer, direction)
                elif command[0] == 'v':
                    puzzle.execute_vertical_rotation(layer, direction)
                elif command[0] == 's':
                    puzzle.execute_lateral_rotation(layer, direction)
            else:
                print("Invalid command format")
                
        except (ValueError, IndexError) as e:
            print(f"Error: {e}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--interactive':
            cube_size = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CUBE_SIZE
            interactive_mode(cube_size)
        elif sys.argv[1] == '--size':
            cube_size = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CUBE_SIZE
            main(cube_size)
        else:
            print("Usage:")
            print("  python puzzle_runner.py                  # Run solver on 3×3×3")
            print("  python puzzle_runner.py --size 4         # Run solver on 4×4×4")
            print("  python puzzle_runner.py --interactive    # Interactive mode (3×3×3)")
            print("  python puzzle_runner.py --interactive 2  # Interactive mode (2×2×2)")
    else:
        main()