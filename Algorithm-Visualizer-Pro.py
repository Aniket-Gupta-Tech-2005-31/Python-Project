import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import heapq
from collections import deque, defaultdict
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Algorithm Visualizer Pro", layout="wide", page_icon="📚")

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
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
    }
    .complexity-badge {
        display: inline-block;
        background-color: #ff6b6b;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
    .success-badge {
        background-color: #51cf66;
    }
    .warning-badge {
        background-color: #fcc419;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📚 Algorithm Visualizer Pro</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; margin-bottom: 3rem;'>
    <h3>An Interactive Learning Platform for Graph Algorithms</h3>
    <p>Learn, visualize, and experiment with 14+ graph algorithms through interactive demonstrations</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_chapter' not in st.session_state:
    st.session_state.current_chapter = 0
if 'quiz_answers' not in st.session_state:
    st.session_state.quiz_answers = {}
if 'user_progress' not in st.session_state:
    st.session_state.user_progress = {
        'completed_chapters': set(),
        'quiz_scores': {},
        'practice_completed': set()
    }

# Sidebar - Table of Contents
st.sidebar.title("📖 Table of Contents")
st.sidebar.markdown("---")

# Algorithm chapters
chapters = [
    {"title": "🏠 Introduction to Graph Algorithms", "icon": "🏠"},
    {"title": "🔍 Search Algorithms", "icon": "🔍"},
    {"title": "🚀 BFS & DFS Fundamentals", "icon": "🚀"},
    {"title": "📍 Dijkstra's Algorithm", "icon": "📍"},
    {"title": "⭐ A* Search Algorithm", "icon": "⭐"},
    {"title": "🔄 Dynamic Programming Algorithms", "icon": "🔄"},
    {"title": "🎯 Advanced Pathfinding", "icon": "🎯"},
    {"title": "📊 Algorithm Comparison", "icon": "📊"},
    {"title": "🏆 Practice & Assessment", "icon": "🏆"}
]

for i, chapter in enumerate(chapters):
    if st.sidebar.button(f"{chapter['icon']} {chapter['title']}", key=f"chap_{i}", 
                        use_container_width=True, type="primary" if st.session_state.current_chapter == i else "secondary"):
        st.session_state.current_chapter = i

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Your Progress")
progress = len(st.session_state.user_progress['completed_chapters']) / len(chapters)
st.sidebar.progress(progress)
st.sidebar.write(f"Completed: {len(st.session_state.user_progress['completed_chapters'])}/{len(chapters)} chapters")

# Chapter 0: Introduction
if st.session_state.current_chapter == 0:
    st.header("🎯 Introduction to Graph Algorithms")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="algorithm-card">
            <h3>What are Graph Algorithms?</h3>
            <p>Graph algorithms are methods for solving problems on graphs, which consist of nodes (vertices) 
            connected by edges. They are fundamental in computer science and have numerous real-world applications.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("📈 Why Learn Graph Algorithms?")
        st.markdown("""
        - **Navigation Systems**: GPS and mapping applications
        - **Social Networks**: Friend recommendations, network analysis
        - **Computer Networks**: Routing protocols, network optimization
        - **Artificial Intelligence**: Pathfinding in games and robotics
        - **Bioinformatics**: Protein interaction networks
        - **Recommendation Systems**: Amazon, Netflix recommendations
        """)
    
    with col2:
        # Create a sample graph visualization
        G = nx.karate_club_graph()
        pos = nx.spring_layout(G, seed=42)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=500, font_size=8, ax=ax, edge_color='gray')
        ax.set_title("Example Graph Structure", fontsize=12)
        st.pyplot(fig)
    
    st.subheader("🎓 Learning Objectives")
    objectives = [
        "Understand fundamental graph traversal algorithms",
        "Learn shortest path algorithms and their applications",
        "Explore heuristic search methods",
        "Compare time and space complexity",
        "Apply algorithms to real-world problems"
    ]
    
    for i, obj in enumerate(objectives):
        st.checkbox(f"**{i+1}.** {obj}", value=(0 in st.session_state.user_progress['completed_chapters']), key=f"obj_{i}")
    
    if st.button("Mark Chapter as Complete", key="complete_0"):
        st.session_state.user_progress['completed_chapters'].add(0)
        st.success("Chapter marked as complete! ✅")
        st.rerun()

# Chapter 1: Search Algorithms
elif st.session_state.current_chapter == 1:
    st.header("🔍 Search Algorithms Fundamentals")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📚 Theory", "🎨 Visualization", "💻 Implementation", "🧩 Practice"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Uninformed vs Informed Search")
            st.markdown("""
            **Uninformed Search (Blind Search):**
            - No additional information about goal
            - Examples: BFS, DFS, Uniform Cost Search
            - Complete but may be inefficient
            
            **Informed Search (Heuristic Search):**
            - Uses domain knowledge (heuristics)
            - Examples: A*, Greedy Best-First
            - More efficient with good heuristics
            """)
            
        with col2:
            st.subheader("Key Properties")
            properties = {
                "Completeness": "Guaranteed to find solution if exists",
                "Optimality": "Finds best possible solution",
                "Time Complexity": "How long it takes to run",
                "Space Complexity": "How much memory it uses"
            }
            
            for prop, desc in properties.items():
                with st.expander(f"📌 {prop}"):
                    st.write(desc)
    
    with tab2:
        st.subheader("Interactive Search Comparison")
        
        # Create a grid for search visualization
        grid_size = st.slider("Grid Size", 5, 15, 10)
        start_pos = (0, 0)
        goal_pos = (grid_size-1, grid_size-1)
        
        algorithm = st.selectbox("Choose Algorithm", ["BFS", "DFS", "Dijkstra", "A*"])
        
        if st.button("Run Search Visualization"):
            # Simplified grid search visualization
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create grid
            grid = np.zeros((grid_size, grid_size))
            
            # Add some obstacles
            for i in range(1, grid_size-1):
                for j in range(1, grid_size-1):
                    if random.random() < 0.2:  # 20% obstacles
                        grid[i][j] = 1
            
            ax.imshow(grid, cmap='Pastel1', interpolation='nearest')
            
            # Mark start and goal
            ax.plot(start_pos[1], start_pos[0], 'go', markersize=15, label='Start')
            ax.plot(goal_pos[1], goal_pos[0], 'ro', markersize=15, label='Goal')
            
            ax.set_title(f"{algorithm} Search Visualization")
            ax.legend()
            st.pyplot(fig)
    
    with tab3:
        st.subheader("BFS Implementation")
        st.code("""
from collections import deque

def bfs(graph, start, goal):
    queue = deque([[start]])
    visited = set([start])
    
    while queue:
        path = queue.popleft()
        node = path[-1]
        
        if node == goal:
            return path
            
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)
    
    return None  # No path found
        """, language="python")
        
        st.subheader("DFS Implementation")
        st.code("""
def dfs(graph, start, goal):
    stack = [[start]]
    visited = set([start])
    
    while stack:
        path = stack.pop()
        node = path[-1]
        
        if node == goal:
            return path
            
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = list(path)
                new_path.append(neighbor)
                stack.append(new_path)
    
    return None
        """, language="python")
    
    with tab4:
        st.subheader("Quick Quiz")
        
        quiz_questions = [
            {
                "question": "Which search algorithm uses a queue?",
                "options": ["BFS", "DFS", "A*", "Dijkstra"],
                "answer": "BFS"
            },
            {
                "question": "What is the time complexity of BFS?",
                "options": ["O(V+E)", "O(V²)", "O(E log V)", "O(V log E)"],
                "answer": "O(V+E)"
            }
        ]
        
        score = 0
        for i, q in enumerate(quiz_questions):
            st.write(f"**Q{i+1}: {q['question']}**")
            user_answer = st.radio(f"Select your answer:", q['options'], key=f"quiz_{i}")
            
            if st.session_state.get(f"quiz_{i}_submitted", False):
                if user_answer == q['answer']:
                    score += 1
        
        if st.button("Submit Quiz"):
            for i in range(len(quiz_questions)):
                st.session_state[f"quiz_{i}_submitted"] = True
            st.session_state.user_progress['quiz_scores'][1] = score
            st.success(f"Quiz submitted! Score: {score}/{len(quiz_questions)}")
            
            if score == len(quiz_questions):
                st.session_state.user_progress['completed_chapters'].add(1)
                st.balloons()

# Chapter 2: BFS & DFS
elif st.session_state.current_chapter == 2:
    st.header("🚀 Breadth-First Search (BFS) & Depth-First Search (DFS)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("BFS - Level Order Traversal")
        st.markdown("""
        **Key Characteristics:**
        - Uses queue data structure
        - Explores all neighbors at current depth first
        - Guarantees shortest path in unweighted graphs
        - Memory intensive for large graphs
        
        **Complexity:**
        """)
        st.markdown('<span class="complexity-badge">Time: O(V + E)</span>', unsafe_allow_html=True)
        st.markdown('<span class="complexity-badge">Space: O(V)</span>', unsafe_allow_html=True)
        
    with col2:
        st.subheader("DFS - Depth First Exploration")
        st.markdown("""
        **Key Characteristics:**
        - Uses stack (recursion)
        - Explores one branch completely before backtracking
        - Memory efficient
        - Doesn't guarantee shortest path
        
        **Complexity:**
        """)
        st.markdown('<span class="complexity-badge">Time: O(V + E)</span>', unsafe_allow_html=True)
        st.markdown('<span class="complexity-badge">Space: O(h)</span>', unsafe_allow_html=True)
    
    # Interactive BFS/DFS visualizer
    st.subheader("🎮 Interactive BFS/DFS Visualizer")
    
    # Graph creation options
    graph_type = st.selectbox("Graph Type", ["Tree", "Grid", "Random Graph"])
    
    if graph_type == "Tree":
        G = nx.balanced_tree(2, 4)  # Binary tree of height 4
    elif graph_type == "Grid":
        G = nx.grid_2d_graph(4, 4)
        G = nx.convert_node_labels_to_integers(G)
    else:
        G = nx.erdos_renyi_graph(15, 0.2, seed=42)
    
    pos = nx.spring_layout(G, seed=42)
    
    algorithm = st.radio("Select Algorithm:", ["BFS", "DFS"], horizontal=True)
    start_node = st.selectbox("Start Node", list(G.nodes()))
    
    if st.button("Start Visualization"):
        # Animation of search process
        placeholder = st.empty()
        
        if algorithm == "BFS":
            visited = set()
            queue = deque([start_node])
            visited.add(start_node)
            
            while queue:
                current = queue.popleft()
                
                # Update visualization
                fig, ax = plt.subplots(figsize=(10, 8))
                nx.draw(G, pos, with_labels=True, node_color='lightblue', ax=ax)
                nx.draw_networkx_nodes(G, pos, nodelist=[current], node_color='red', ax=ax)
                nx.draw_networkx_nodes(G, pos, nodelist=list(visited), node_color='yellow', ax=ax)
                
                placeholder.pyplot(fig)
                plt.close(fig)
                time.sleep(0.5)
                
                for neighbor in G.neighbors(current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
        
        else:  # DFS
            visited = set()
            stack = [start_node]
            
            while stack:
                current = stack.pop()
                
                if current not in visited:
                    visited.add(current)
                    
                    # Update visualization
                    fig, ax = plt.subplots(figsize=(10, 8))
                    nx.draw(G, pos, with_labels=True, node_color='lightblue', ax=ax)
                    nx.draw_networkx_nodes(G, pos, nodelist=[current], node_color='red', ax=ax)
                    nx.draw_networkx_nodes(G, pos, nodelist=list(visited), node_color='yellow', ax=ax)
                    
                    placeholder.pyplot(fig)
                    plt.close(fig)
                    time.sleep(0.5)
                    
                    for neighbor in reversed(list(G.neighbors(current))):
                        if neighbor not in visited:
                            stack.append(neighbor)

# Chapter 3: Dijkstra's Algorithm
elif st.session_state.current_chapter == 3:
    st.header("📍 Dijkstra's Algorithm - Single Source Shortest Path")
    
    st.markdown("""
    <div class="algorithm-card">
        <h3>Algorithm Overview</h3>
        <p>Dijkstra's algorithm finds the shortest path from a source node to all other nodes 
        in a weighted graph with <strong>non-negative</strong> edge weights.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Step-by-Step Process")
        steps = [
            "1. Initialize distances: 0 for source, ∞ for all other nodes",
            "2. Create a priority queue with all nodes",
            "3. While queue not empty:",
            "   - Extract node u with minimum distance",
            "   - For each neighbor v of u:",
            "     - Calculate alternative distance = dist[u] + weight(u,v)",
            "     - If alternative < dist[v], update dist[v] and set u as predecessor"
        ]
        
        for step in steps:
            st.write(step)
    
    with col2:
        st.subheader("Properties")
        st.markdown('<span class="complexity-badge success-badge">Optimal for non-negative weights</span>', unsafe_allow_html=True)
        st.markdown('<span class="complexity-badge">Time: O((V+E) log V)</span>', unsafe_allow_html=True)
        st.markdown('<span class="complexity-badge">Space: O(V)</span>', unsafe_allow_html=True)
        st.markdown('<span class="complexity-badge warning-badge">Fails with negative weights</span>', unsafe_allow_html=True)
    
    # Dijkstra visualization
    st.subheader("🎨 Dijkstra's Algorithm Visualizer")
    
    # Create weighted graph
    G = nx.Graph()
    weighted_edges = [
        (0, 1, 4), (0, 2, 2), (1, 2, 1), (1, 3, 5),
        (2, 3, 8), (2, 4, 10), (3, 4, 2), (3, 5, 6),
        (4, 5, 3)
    ]
    G.add_weighted_edges_from(weighted_edges)
    pos = nx.spring_layout(G, seed=42)
    
    start_node = st.selectbox("Source Node", list(G.nodes()))
    
    if st.button("Run Dijkstra"):
        # Dijkstra implementation with visualization
        distances = {node: float('inf') for node in G.nodes()}
        distances[start_node] = 0
        predecessors = {}
        visited = set()
        
        priority_queue = [(0, start_node)]
        
        placeholder = st.empty()
        
        while priority_queue:
            current_dist, current_node = heapq.heappop(priority_queue)
            
            if current_node in visited:
                continue
                
            visited.add(current_node)
            
            # Visualization update
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Draw graph
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', ax=ax, node_size=800)
            nx.draw_networkx_edges(G, pos, ax=ax)
            nx.draw_networkx_labels(G, pos, ax=ax)
            
            # Highlight visited nodes
            nx.draw_networkx_nodes(G, pos, nodelist=list(visited), node_color='yellow', ax=ax, node_size=800)
            nx.draw_networkx_nodes(G, pos, nodelist=[current_node], node_color='red', ax=ax, node_size=800)
            
            # Draw edge weights
            edge_labels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
            
            # Display current distances
            distance_text = "\n".join([f"Node {node}: {dist}" for node, dist in distances.items()])
            ax.text(1.1, 0.5, f"Current Distances:\n{distance_text}", 
                   transform=ax.transAxes, fontsize=10, verticalalignment='center',
                   bbox=dict(boxstyle="round", facecolor="wheat"))
            
            ax.set_title(f"Dijkstra's Algorithm - Current Node: {current_node}, Distance: {current_dist}")
            ax.axis('off')
            
            placeholder.pyplot(fig)
            plt.close(fig)
            time.sleep(1)
            
            for neighbor in G.neighbors(current_node):
                if neighbor not in visited:
                    new_dist = current_dist + G[current_node][neighbor]['weight']
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        predecessors[neighbor] = current_node
                        heapq.heappush(priority_queue, (new_dist, neighbor))
        
        st.success("Algorithm completed! Final distances calculated.")
        st.write("**Final Distances:**", distances)

# Chapter 4: A* Algorithm
elif st.session_state.current_chapter == 4:
    st.header("⭐ A* Search Algorithm")
    
    st.markdown("""
    <div class="algorithm-card">
        <h3>A* Algorithm Overview</h3>
        <p>A* is an informed search algorithm that finds the shortest path between nodes using 
        heuristics to guide its search. It combines the strengths of Dijkstra's algorithm and 
        greedy best-first search.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Key Formula: f(n) = g(n) + h(n)")
        st.markdown("""
        - **g(n)**: Actual cost from start node to current node n
        - **h(n)**: Heuristic estimate from node n to goal
        - **f(n)**: Estimated total cost through node n
        
        **Common Heuristics:**
        - Manhattan distance (grids)
        - Euclidean distance
        - Chebyshev distance
        """)
    
    with col2:
        st.subheader("Properties")
        st.markdown('<span class="complexity-badge success-badge">Optimal with admissible heuristic</span>', unsafe_allow_html=True)
        st.markdown('<span class="complexity-badge">Time: O(b^d)</span>', unsafe_allow_html=True)
        st.markdown('<span class="complexity-badge">Space: O(b^d)</span>', unsafe_allow_html=True)
        st.markdown('<span class="complexity-badge warning-badge">Heuristic quality matters</span>', unsafe_allow_html=True)

# Chapter 5: Dynamic Programming Algorithms
elif st.session_state.current_chapter == 5:
    st.header("🔄 Dynamic Programming Algorithms")
    
    tab1, tab2 = st.tabs(["Bellman-Ford", "Floyd-Warshall"])
    
    with tab1:
        st.subheader("Bellman-Ford Algorithm")
        st.markdown("""
        **Purpose:** Finds shortest paths from single source in graphs with negative weights
        **Key Feature:** Can detect negative weight cycles
        **Time Complexity:** O(VE)
        """)
        
        st.code("""
def bellman_ford(graph, source):
    distance = {node: float('inf') for node in graph}
    distance[source] = 0
    
    # Relax edges V-1 times
    for _ in range(len(graph) - 1):
        for u, v, weight in graph.edges(data='weight'):
            if distance[u] + weight < distance[v]:
                distance[v] = distance[u] + weight
    
    # Check for negative cycles
    for u, v, weight in graph.edges(data='weight'):
        if distance[u] + weight < distance[v]:
            return None  # Negative cycle detected
    
    return distance
        """, language="python")
    
    with tab2:
        st.subheader("Floyd-Warshall Algorithm")
        st.markdown("""
        **Purpose:** Finds shortest paths between all pairs of nodes
        **Key Feature:** Works with negative weights (no negative cycles)
        **Time Complexity:** O(V³)
        """)

# Chapter 6: Advanced Pathfinding
elif st.session_state.current_chapter == 6:
    st.header("🎯 Advanced Pathfinding Algorithms")
    
    algorithms = [
        ("Bidirectional Search", "Searches from both start and goal simultaneously"),
        ("Jump Point Search", "Optimizes A* for uniform-cost grids"),
        ("RRT", "Rapidly-exploring Random Tree for high-dimensional spaces"),
        ("PRM", "Probabilistic Roadmap Method for motion planning")
    ]
    
    for algo, desc in algorithms:
        with st.expander(f"📌 {algo}"):
            st.write(desc)
            st.markdown('<span class="complexity-badge">Advanced</span>', unsafe_allow_html=True)

# Chapter 7: Algorithm Comparison
elif st.session_state.current_chapter == 7:
    st.header("📊 Algorithm Comparison & Analysis")
    
    # Comparative analysis
    algorithms_data = [
        {"Algorithm": "BFS", "Time": "O(V+E)", "Space": "O(V)", "Optimal": "Yes", "Complete": "Yes", "Weights": "No"},
        {"Algorithm": "DFS", "Time": "O(V+E)", "Space": "O(h)", "Optimal": "No", "Complete": "Yes", "Weights": "No"},
        {"Algorithm": "Dijkstra", "Time": "O((V+E)logV)", "Space": "O(V)", "Optimal": "Yes", "Complete": "Yes", "Weights": "Non-negative"},
        {"Algorithm": "A*", "Time": "O(b^d)", "Space": "O(b^d)", "Optimal": "Yes*", "Complete": "Yes", "Weights": "Yes"},
        {"Algorithm": "Bellman-Ford", "Time": "O(VE)", "Space": "O(V)", "Optimal": "Yes", "Complete": "Yes", "Weights": "Any"},
    ]
    
    st.table(algorithms_data)
    
    # Performance comparison chart
    st.subheader("Performance Comparison")
    
    # Sample performance data (in milliseconds)
    nodes_range = [10, 50, 100, 200]
    bfs_times = [1, 5, 12, 25]
    dfs_times = [1, 4, 10, 22]
    dijkstra_times = [2, 8, 20, 45]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(nodes_range, bfs_times, label='BFS', linewidth=3)
    ax.plot(nodes_range, dfs_times, label='DFS', linewidth=3)
    ax.plot(nodes_range, dijkstra_times, label='Dijkstra', linewidth=3)
    
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Execution Time (ms)')
    ax.set_title('Algorithm Performance Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)

# Chapter 8: Practice & Assessment
elif st.session_state.current_chapter == 8:
    st.header("🏆 Practice & Assessment")
    
    tab1, tab2, tab3 = st.tabs(["🧠 Challenge Problems", "📝 Final Assessment", "🏅 Achievement Board"])
    
    with tab1:
        st.subheader("Algorithm Challenges")
        
        challenges = [
            {
                "title": "Maze Solver",
                "description": "Implement BFS to solve a maze",
                "difficulty": "⭐",
                "completed": 1 in st.session_state.user_progress['practice_completed']
            },
            {
                "title": "Shortest Path Network",
                "description": "Use Dijkstra to find optimal routes",
                "difficulty": "⭐⭐",
                "completed": 2 in st.session_state.user_progress['practice_completed']
            },
            {
                "title": "A* with Custom Heuristics",
                "description": "Implement A* with different heuristic functions",
                "difficulty": "⭐⭐⭐",
                "completed": 3 in st.session_state.user_progress['practice_completed']
            }
        ]
        
        for i, challenge in enumerate(challenges):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"**{challenge['title']}**")
                st.write(challenge['description'])
            with col2:
                st.write(challenge['difficulty'])
            with col3:
                if challenge['completed']:
                    st.success("✅ Completed")
                else:
                    if st.button("Start", key=f"challenge_{i}"):
                        st.info(f"Starting {challenge['title']}...")
                        # Add challenge implementation here
    
    with tab2:
        st.subheader("Final Assessment")
        
        if len(st.session_state.user_progress['completed_chapters']) >= 7:
            st.success("🎉 You can now take the final assessment!")
            
            final_questions = [
                {
                    "question": "Which algorithm is best for unweighted graphs?",
                    "options": ["Dijkstra", "BFS", "A*", "Bellman-Ford"],
                    "answer": "BFS"
                },
                {
                    "question": "What is the main advantage of A* over Dijkstra?",
                    "options": ["Faster with good heuristics", "Handles negative weights", "Uses less memory", "Always optimal"],
                    "answer": "Faster with good heuristics"
                }
            ]
            
            final_score = 0
            for i, q in enumerate(final_questions):
                st.write(f"**Q{i+1}: {q['question']}**")
                user_answer = st.radio(f"Select your answer:", q['options'], key=f"final_{i}")
                
                if st.session_state.get(f"final_{i}_submitted", False):
                    if user_answer == q['answer']:
                        final_score += 1
            
            if st.button("Submit Final Assessment"):
                for i in range(len(final_questions)):
                    st.session_state[f"final_{i}_submitted"] = True
                st.session_state.user_progress['quiz_scores']['final'] = final_score
                st.success(f"Final assessment submitted! Score: {final_score}/{len(final_questions)}")
                
                if final_score == len(final_questions):
                    st.session_state.user_progress['completed_chapters'].add(8)
                    st.balloons()
        else:
            st.warning("Complete at least 7 chapters to unlock the final assessment!")
    
    with tab3:
        st.subheader("Your Achievements")
        
        achievements = [
            {"name": "First Steps", "description": "Complete Introduction", "earned": 0 in st.session_state.user_progress['completed_chapters']},
            {"name": "Search Master", "description": "Complete Search Algorithms", "earned": 1 in st.session_state.user_progress['completed_chapters']},
            {"name": "Pathfinder", "description": "Complete all pathfinding algorithms", "earned": {2,3,4}.issubset(st.session_state.user_progress['completed_chapters'])},
            {"name": "Algorithm Expert", "description": "Complete all chapters", "earned": len(st.session_state.user_progress['completed_chapters']) == len(chapters)}
        ]
        
        for achievement in achievements:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{achievement['name']}**")
                st.write(achievement['description'])
            with col2:
                if achievement['earned']:
                    st.success("🏆 Earned")
                else:
                    st.info("🔒 Locked")

# Footer with navigation
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    if st.session_state.current_chapter > 0:
        if st.button("⬅️ Previous Chapter"):
            st.session_state.current_chapter -= 1
            st.rerun()

with col2:
    progress = (st.session_state.current_chapter + 1) / len(chapters)
    st.progress(progress)
    st.write(f"Chapter {st.session_state.current_chapter + 1} of {len(chapters)}")

with col3:
    if st.session_state.current_chapter < len(chapters) - 1:
        if st.button("Next Chapter ➡️"):
            st.session_state.current_chapter += 1
            st.rerun()
    else:
        if st.button("🎉 Complete Course"):
            st.balloons()
            st.success("Congratulations on completing the Algorithm Visualizer Pro course!")

# Add download certificate option
if len(st.session_state.user_progress['completed_chapters']) == len(chapters):
    st.sidebar.markdown("---")
    if st.sidebar.button("📜 Download Certificate"):
        st.sidebar.success("Certificate generated! (Feature in development)")