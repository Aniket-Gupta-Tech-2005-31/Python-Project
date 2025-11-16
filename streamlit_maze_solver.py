import streamlit as st
import numpy as np
import heapq
import random
import time
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque

# Page configuration
st.set_page_config(
    page_title="Maze Solver Pro", 
    page_icon="🧩", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .algorithm-card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-bottom: 1rem;
        border-left: 5px solid #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    .success-metric {
        border-left: 5px solid #28a745;
    }
    .warning-metric {
        border-left: 5px solid #ffc107;
    }
    .info-metric {
        border-left: 5px solid #17a2b8;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["Maze Solver", "Algorithm Info", "Performance Comparison"])

# ------------------------------ Enhanced Helper Functions ------------------------------ #
def generate_maze_with_difficulty(n, difficulty="medium"):
    """Generate maze with configurable difficulty"""
    if difficulty == "easy":
        p = [0.85, 0.15]  # More open space
    elif difficulty == "medium":
        p = [0.75, 0.25]
    elif difficulty == "hard":
        p = [0.65, 0.35]  # More obstacles
    else:  # random
        p = [0.7, 0.3]
    
    maze = np.random.choice([0, 1], size=(n, n), p=p)
    maze[0][0] = 0  # Ensure start is open
    maze[-1][-1] = 0  # Ensure goal is open
    
    # Ensure there's at least one path
    if not has_path(maze, (0, 0), (n-1, n-1)):
        return generate_maze_with_difficulty(n, difficulty)
    
    return maze

def has_path(maze, start, goal):
    """Check if there's a path from start to goal"""
    n = len(maze)
    visited = set()
    stack = [start]
    
    while stack:
        cell = stack.pop()
        if cell == goal:
            return True
        if cell in visited:
            continue
        visited.add(cell)
        
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = cell[0] + dx, cell[1] + dy
            if 0 <= nx < n and 0 <= ny < n and maze[nx][ny] == 0 and (nx, ny) not in visited:
                stack.append((nx, ny))
    
    return False

def display_maze_enhanced(maze, path=None, explored=None, current=None, delay=0.01):
    """Enhanced maze display with better visualization"""
    if path is None:
        path = []
    if explored is None:
        explored = set()
        
    n = len(maze)
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create color grid
    grid = np.zeros((n, n, 3))
    
    # Set colors
    for i in range(n):
        for j in range(n):
            if (i, j) == current:
                grid[i, j] = [1, 0.5, 0]  # Orange for current
            elif (i, j) in path:
                grid[i, j] = [0, 1, 0]  # Green for path
            elif (i, j) in explored:
                grid[i, j] = [0.5, 0.7, 1]  # Light blue for explored
            elif maze[i][j] == 1:
                grid[i, j] = [0.2, 0.2, 0.2]  # Dark gray for walls
            else:
                grid[i, j] = [1, 1, 1]  # White for empty
    
    ax.imshow(grid)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Mark start and goal
    ax.text(0, 0, "S", ha='center', va='center', fontsize=20, fontweight='bold', color='red')
    ax.text(n-1, n-1, "G", ha='center', va='center', fontsize=20, fontweight='bold', color='red')
    
    plt.tight_layout()
    maze_display.pyplot(fig)
    plt.close(fig)
    
    if delay > 0:
        time.sleep(delay)

def display_maze_simple(maze, path=None, explored=None, delay=0.01):
    """Simple text-based display as fallback"""
    if path is None:
        path = []
    if explored is None:
        explored = set()
        
    grid_str = ""
    n = len(maze)
    for i in range(n):
        for j in range(n):
            if (i, j) == (0, 0):
                grid_str += "🚶"  # Start
            elif (i, j) == (n-1, n-1):
                grid_str += "🏁"  # Goal
            elif (i, j) in path:
                grid_str += "🟩"  # Path
            elif (i, j) in explored:
                grid_str += "🟦"  # Explored
            elif maze[i][j] == 1:
                grid_str += "⬛"  # Wall
            else:
                grid_str += "⬜"  # Empty
        grid_str += "\n"
    maze_display.text(grid_str)
    if delay > 0:
        time.sleep(delay)

# Generate neighbors
def neighbors(cell, n):
    x, y = cell
    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
        nx, ny = x+dx, y+dy
        if 0 <= nx < n and 0 <= ny < n:
            yield (nx, ny)

# ------------------------------ Enhanced Algorithms with Metrics ------------------------------ #
def bfs(maze, start, goal, display_func, delay):
    n = len(maze)
    queue = deque([start])
    parent = {start: None}
    explored = set([start])
    nodes_explored = 0
    
    while queue:
        cell = queue.popleft()
        nodes_explored += 1
        
        if cell == goal:
            break
            
        for nb in neighbors(cell, n):
            if maze[nb[0]][nb[1]] == 0 and nb not in explored:
                explored.add(nb)
                parent[nb] = cell
                queue.append(nb)
        
        # Call display function with correct arguments
        if display_func == display_maze_enhanced:
            display_func(maze, [], explored, cell, delay)
        else:
            display_func(maze, [], explored, delay)
    
    path = []
    cur = goal
    while cur:
        path.append(cur)
        cur = parent.get(cur)
    
    return path[::-1], len(explored), nodes_explored

def dfs(maze, start, goal, display_func, delay):
    n = len(maze)
    stack = [start]
    parent = {start: None}
    explored = set()
    nodes_explored = 0
    
    while stack:
        cell = stack.pop()
        if cell in explored:
            continue
            
        explored.add(cell)
        nodes_explored += 1
        
        if cell == goal:
            break
            
        for nb in neighbors(cell, n):
            if maze[nb[0]][nb[1]] == 0 and nb not in explored:
                parent[nb] = cell
                stack.append(nb)
        
        # Call display function with correct arguments
        if display_func == display_maze_enhanced:
            display_func(maze, [], explored, cell, delay)
        else:
            display_func(maze, [], explored, delay)
    
    path = []
    cur = goal
    while cur:
        path.append(cur)
        cur = parent.get(cur)
    
    return path[::-1], len(explored), nodes_explored

def dijkstra(maze, start, goal, display_func, delay):
    n = len(maze)
    pq = [(0, start)]
    dist = {start: 0}
    parent = {start: None}
    explored = set()
    nodes_explored = 0
    
    while pq:
        d, cell = heapq.heappop(pq)
        if cell in explored:
            continue
            
        explored.add(cell)
        nodes_explored += 1
        
        if cell == goal:
            break
            
        for nb in neighbors(cell, n):
            if maze[nb[0]][nb[1]] == 0:
                nd = d + 1
                if nd < dist.get(nb, float('inf')):
                    dist[nb] = nd
                    parent[nb] = cell
                    heapq.heappush(pq, (nd, nb))
        
        # Call display function with correct arguments
        if display_func == display_maze_enhanced:
            display_func(maze, [], explored, cell, delay)
        else:
            display_func(maze, [], explored, delay)
    
    path = []
    cur = goal
    while cur:
        path.append(cur)
        cur = parent.get(cur)
    
    return path[::-1], len(explored), nodes_explored

def astar(maze, start, goal, display_func, delay):
    n = len(maze)
    h = lambda x: abs(x[0]-goal[0]) + abs(x[1]-goal[1])  # Manhattan distance
    pq = [(h(start), 0, start)]
    parent = {start: None}
    g = {start: 0}
    explored = set()
    nodes_explored = 0
    
    while pq:
        f, cost, cell = heapq.heappop(pq)
        if cell in explored:
            continue
            
        explored.add(cell)
        nodes_explored += 1
        
        if cell == goal:
            break
            
        for nb in neighbors(cell, n):
            if maze[nb[0]][nb[1]] == 0:
                ng = g[cell] + 1
                if ng < g.get(nb, float('inf')):
                    g[nb] = ng
                    parent[nb] = cell
                    heapq.heappush(pq, (ng + h(nb), ng, nb))
        
        # Call display function with correct arguments
        if display_func == display_maze_enhanced:
            display_func(maze, [], explored, cell, delay)
        else:
            display_func(maze, [], explored, delay)
    
    path = []
    cur = goal
    while cur:
        path.append(cur)
        cur = parent.get(cur)
    
    return path[::-1], len(explored), nodes_explored

# ------------------------------ UI Layout ------------------------------ #
if page == "Maze Solver":
    st.markdown('<h1 class="main-header">🧩 Interactive Maze Solver Pro</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Configuration")
        
        grid_size = st.slider("Grid Size", 5, 30, 15)
        difficulty = st.selectbox("Maze Difficulty", ["easy", "medium", "hard", "random"])
        speed = st.slider("Animation Speed", 0.001, 0.5, 0.05)
        display_type = st.radio("Display Type", ["Visual", "Text"])
        
        algo_name = st.selectbox("Choose Algorithm", [
            "BFS", "DFS", "Dijkstra", "A*"
        ])
        
        if st.button("🎯 Generate New Maze"):
            st.session_state.maze = generate_maze_with_difficulty(grid_size, difficulty)
        
        if "maze" not in st.session_state:
            st.session_state.maze = generate_maze_with_difficulty(grid_size, difficulty)
    
    with col2:
        st.subheader("🗺️ Maze Visualization")
        maze_display = st.empty()
        
        # Display the current maze
        if display_type == "Visual":
            display_maze_enhanced(st.session_state.maze, delay=0)
        else:
            display_maze_simple(st.session_state.maze, delay=0)
    
    # Solve button
    if st.button("🚀 Solve Maze", type="primary"):
        start, goal = (0, 0), (grid_size-1, grid_size-1)
        
        algos = {
            "BFS": bfs, 
            "DFS": dfs, 
            "Dijkstra": dijkstra, 
            "A*": astar
        }
        
        algo_func = algos[algo_name]
        display_func = display_maze_enhanced if display_type == "Visual" else display_maze_simple
        
        # Solve with progress
        with st.spinner(f"Solving with {algo_name}..."):
            start_time = time.time()
            path, explored_count, nodes_processed = algo_func(
                st.session_state.maze, start, goal, display_func, speed
            )
            end_time = time.time()
        
        # Display final path
        if display_type == "Visual":
            display_maze_enhanced(st.session_state.maze, path, set(), delay=0)
        else:
            display_maze_simple(st.session_state.maze, path, set(), delay=0)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'<div class="metric-card success-metric" style="color:black;">'
                       f'<h3>⏱️</h3><h4>{end_time - start_time:.3f}s</h4><p>Time</p></div>', 
                       unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'<div class="metric-card info-metric" style="color:black;">'
                       f'<h3>📏</h3><h4>{len(path)}</h4><p>Path Length</p></div>', 
                       unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'<div class="metric-card warning-metric" style="color:black;">'
                       f'<h3>🔍</h3><h4>{explored_count}</h4><p>Cells Explored</p></div>', 
                       unsafe_allow_html=True)
        
        with col4:
            st.markdown(f'<div class="metric-card info-metric" style="color:black;">'
                       f'<h3>🔄</h3><h4>{nodes_processed}</h4><p>Nodes Processed</p></div>', 
                       unsafe_allow_html=True)
        
        if path and path[-1] == goal:
            st.success(f"✅ {algo_name} found a path in {end_time - start_time:.3f} seconds!")
        else:
            st.error(f"❌ {algo_name} couldn't find a path to the goal!")

elif page == "Algorithm Info":
    st.markdown('<h1 class="main-header">📘 Algorithm & Code Explanation</h1>', unsafe_allow_html=True)
    
    algo = st.selectbox("Select Algorithm", [
        "BFS", "DFS", "Dijkstra", "A*"
    ])
    
    tab1, tab2, tab3, tab4 = st.tabs(["📖 Overview", "🧠 How It Works", "⚖️ Pros & Cons", "💻 Implementation"])
    
    # Algorithm information database
    algo_info = {
        "BFS": {
            "overview": "Breadth-First Search (BFS) is a graph traversal algorithm that explores all neighbor nodes at the present depth before moving on to nodes at the next depth level. It's guaranteed to find the shortest path in unweighted graphs.",
            "how_it_works": "1. Start from the initial node\n2. Explore all neighbors at the current depth\n3. Move to the next level of neighbors\n4. Use a queue to manage the order of exploration\n5. Stop when the goal is found",
            "pros": ["Finds shortest path in unweighted graphs", "Complete (will find solution if exists)", "Simple to implement"],
            "cons": ["Memory intensive for large graphs", "Not optimal for weighted graphs", "Can be slow for deep graphs"],
            "complexity": "Time: O(V + E), Space: O(V)",
            "use_cases": ["Shortest path in unweighted graphs", "Web crawling", "Social network analysis"]
        },
        "DFS": {
            "overview": "Depth-First Search (DFS) explores as far as possible along each branch before backtracking. It uses a stack (either explicitly or via recursion) to keep track of nodes to visit.",
            "how_it_works": "1. Start from the initial node\n2. Explore one branch as deep as possible\n3. Backtrack when no more unexplored nodes\n4. Use a stack to manage the order of exploration",
            "pros": ["Memory efficient (O(depth))", "Can find solutions faster in some cases", "Good for maze generation"],
            "cons": ["Not guaranteed to find shortest path", "Can get stuck in deep branches", "May not find solution in infinite graphs"],
            "complexity": "Time: O(V + E), Space: O(V)",
            "use_cases": ["Maze generation", "Topological sorting", "Solving puzzles"]
        },
        "Dijkstra": {
            "overview": "Dijkstra's algorithm finds the shortest path between nodes in a graph with non-negative edge weights. It uses a priority queue to always expand the most promising node.",
            "how_it_works": "1. Assign tentative distance values to all nodes\n2. Set initial node distance to 0, others to infinity\n3. Visit unvisited node with smallest tentative distance\n4. Update neighbors' distances if shorter path found\n5. Mark current node as visited",
            "pros": ["Guaranteed optimal for non-negative weights", "Efficient with priority queue", "Widely applicable"],
            "cons": ["Fails with negative weights", "Can be slow for large graphs", "Requires priority queue implementation"],
            "complexity": "Time: O((V+E) log V), Space: O(V)",
            "use_cases": ["Network routing protocols", "GPS navigation", "Robotics path planning"]
        },
        "A*": {
            "overview": "A* search algorithm finds the shortest path by combining the actual cost from start (g) with a heuristic estimate to goal (h). It's often more efficient than Dijkstra for single-pair shortest path.",
            "how_it_works": "1. Evaluate nodes using f(n) = g(n) + h(n)\n2. g(n) = actual cost from start to n\n3. h(n) = heuristic estimate from n to goal\n4. Always expand node with lowest f(n) first\n5. Use admissible heuristic for optimality",
            "pros": ["Optimal with admissible heuristic", "Generally faster than Dijkstra", "Flexible with different heuristics"],
            "cons": ["Memory intensive", "Heuristic quality affects performance", "Not optimal with inconsistent heuristic"],
            "complexity": "Time: O(b^d), Space: O(b^d)",
            "use_cases": ["Game AI pathfinding", "Robotics", "Natural language processing"]
        }
    }
    
    info = algo_info.get(algo, {})
    
    with tab1:
        st.header("📖 Overview")
        st.markdown(f'<div class="algorithm-card" style="color:black">{info.get("overview", "")}</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Complexity")
            st.info(f"**Time:** {info.get('complexity', '').split(', ')[0]}")
            st.info(f"**Space:** {info.get('complexity', '').split(', ')[1]}")
        
        with col2:
            st.subheader("🎯 Use Cases")
            for use_case in info.get("use_cases", []):
                st.write(f"• {use_case}")
    
    with tab2:
        st.header("🧠 How It Works")
        st.write(info.get("how_it_works", ""))
        
        # Add visual explanation
        if algo == "BFS":
            st.image("https://upload.wikimedia.org/wikipedia/commons/5/5d/Breadth-First-Search-Algorithm.gif", 
                    caption="BFS explores level by level")
        elif algo == "DFS":
            st.image("https://upload.wikimedia.org/wikipedia/commons/7/7f/Depth-First-Search.gif", 
                    caption="DFS goes deep before wide")
        elif algo == "Dijkstra":
            st.image("https://upload.wikimedia.org/wikipedia/commons/5/57/Dijkstra_Animation.gif", 
                    caption="Dijkstra expands shortest path first")
        elif algo == "A*":
            st.image("https://upload.wikimedia.org/wikipedia/commons/5/5d/AstarExampleEn.gif", 
                    caption="A* uses heuristic to guide search")
    
    with tab3:
        st.header("⚖️ Pros & Cons")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("✅ Advantages")
            for pro in info.get("pros", []):
                st.success(pro)
        
        with col2:
            st.subheader("❌ Limitations")
            for con in info.get("cons", []):
                st.error(con)
    
    with tab4:
        st.header("💻 Implementation")
        
        if algo == "BFS":
            code = """
def bfs(maze, start, goal):
    from collections import deque
    queue = deque([start])
    visited = set([start])
    parent = {start: None}
    
    while queue:
        current = queue.popleft()
        
        if current == goal:
            break
            
        for neighbor in get_neighbors(current, maze):
            if neighbor not in visited and maze[neighbor[0]][neighbor[1]] == 0:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)
    
    return reconstruct_path(parent, start, goal)
"""
        elif algo == "DFS":
            code = """
def dfs(maze, start, goal):
    stack = [start]
    visited = set()
    parent = {start: None}
    
    while stack:
        current = stack.pop()
        
        if current in visited:
            continue
        visited.add(current)
        
        if current == goal:
            break
            
        for neighbor in get_neighbors(current, maze):
            if neighbor not in visited and maze[neighbor[0]][neighbor[1]] == 0:
                parent[neighbor] = current
                stack.append(neighbor)
    
    return reconstruct_path(parent, start, goal)
"""
        elif algo == "Dijkstra":
            code = """
def dijkstra(maze, start, goal):
    import heapq
    pq = [(0, start)]
    distances = {start: 0}
    parent = {start: None}
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if current == goal:
            break
            
        if current_dist > distances.get(current, float('inf')):
            continue
            
        for neighbor in get_neighbors(current, maze):
            if maze[neighbor[0]][neighbor[1]] == 0:
                new_dist = current_dist + 1
                if new_dist < distances.get(neighbor, float('inf')):
                    distances[neighbor] = new_dist
                    parent[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))
    
    return reconstruct_path(parent, start, goal)
"""
        elif algo == "A*":
            code = """
def astar(maze, start, goal):
    import heapq
    def heuristic(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])  # Manhattan distance
    
    pq = [(heuristic(start, goal), 0, start)]
    g_costs = {start: 0}
    parent = {start: None}
    
    while pq:
        f, g, current = heapq.heappop(pq)
        
        if current == goal:
            break
            
        if g > g_costs.get(current, float('inf')):
            continue
            
        for neighbor in get_neighbors(current, maze):
            if maze[neighbor[0]][neighbor[1]] == 0:
                new_g = g + 1
                if new_g < g_costs.get(neighbor, float('inf')):
                    g_costs[neighbor] = new_g
                    parent[neighbor] = current
                    heapq.heappush(pq, (new_g + heuristic(neighbor, goal), new_g, neighbor))
    
    return reconstruct_path(parent, start, goal)
"""
        
        st.code(code, language="python")

elif page == "Performance Comparison":
    st.markdown('<h1 class="main-header">📊 Algorithm Performance Comparison</h1>', unsafe_allow_html=True)
    
    st.write("Compare the performance of different algorithms across various maze configurations.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Test Maze Size", 5, 20, 10)
        num_tests = st.slider("Number of Tests", 1, 10, 5)
    
    with col2:
        difficulty = st.selectbox("Test Difficulty", ["easy", "medium", "hard"])
    
    if st.button("🏃 Run Performance Tests"):
        results = []
        algorithms = ["BFS", "DFS", "Dijkstra", "A*"]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, algo_name in enumerate(algorithms):
            status_text.text(f"Testing {algo_name}...")
            
            times = []
            path_lengths = []
            explored_counts = []
            
            for test in range(num_tests):
                maze = generate_maze_with_difficulty(test_size, difficulty)
                start, goal = (0, 0), (test_size-1, test_size-1)
                
                algos = {
                    "BFS": bfs, 
                    "DFS": dfs, 
                    "Dijkstra": dijkstra, 
                    "A*": astar
                }
                
                algo_func = algos[algo_name]
                
                start_time = time.time()
                path, explored, _ = algo_func(maze, start, goal, lambda *args: None, 0)
                end_time = time.time()
                
                times.append(end_time - start_time)
                path_lengths.append(len(path) if path and path[-1] == goal else 0)
                explored_counts.append(explored)
            
            results.append({
                "Algorithm": algo_name,
                "Avg Time (s)": np.mean(times),
                "Avg Path Length": np.mean(path_lengths),
                "Avg Cells Explored": np.mean(explored_counts),
                "Success Rate": np.mean([1 if pl > 0 else 0 for pl in path_lengths]) * 100
            })
            
            progress_bar.progress((i + 1) / len(algorithms))
        
        status_text.text("Complete!")
        
        # Display results
        df = pd.DataFrame(results)
        st.subheader("📈 Performance Results")
        st.dataframe(df.style.format({
            "Avg Time (s)": "{:.4f}",
            "Avg Path Length": "{:.1f}",
            "Avg Cells Explored": "{:.1f}",
            "Success Rate": "{:.1f}%"
        }))
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⏱️ Average Execution Time")
            chart_data = df[["Algorithm", "Avg Time (s)"]].set_index("Algorithm")
            st.bar_chart(chart_data)
        
        with col2:
            st.subheader("🔍 Average Cells Explored")
            chart_data = df[["Algorithm", "Avg Cells Explored"]].set_index("Algorithm")
            st.bar_chart(chart_data)
        
        # Insights
        st.subheader("💡 Key Insights")
        best_time = df.loc[df["Avg Time (s)"].idxmin()]
        best_path = df.loc[df["Avg Path Length"].idxmin()]
        
        st.info(f"**Fastest Algorithm:** {best_time['Algorithm']} ({best_time['Avg Time (s)']:.4f}s)")
        st.info(f"**Shortest Paths:** {best_path['Algorithm']} ({best_path['Avg Path Length']:.1f} cells)")
        st.info(f"**Most Efficient Explorer:** {df.loc[df['Avg Cells Explored'].idxmin()]['Algorithm']}")