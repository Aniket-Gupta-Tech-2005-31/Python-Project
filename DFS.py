graph = {
    'A': set(['B', 'C']),
    'B': set(['A', 'D', 'E']),
    'C': set(['A', 'F']),
    'D': set(['B']),
    'E': set(['B', 'F']),
    'F': set(['C', 'E'])
}

def dfs_recursive(graph, node, visited=None):
    if visited is None:
        visited = []
    if node not in visited:
        visited.append(node)
        for neighbour in sorted(graph[node]):
            dfs_recursive(graph, neighbour, visited)
    return visited

visited = dfs_recursive(graph, 'A')
print("Recursive DFS order:", visited)
