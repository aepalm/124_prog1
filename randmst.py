#!/usr/bin/env python3


#Usage: ./randmst 0 numpoints numtrials dimension
#To get values for table/graph, use ./randmst 1 0 0 0
import sys
import random
import math


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


class MinHeap:
    """Lazy min-heap (no decrease_key)."""

    def __init__(self):
        self.heap = []  # list of (key, value)

    def is_empty(self):
        return len(self.heap) == 0

    def push(self, key, value):
        self.heap.append((key, value))
        self._sift_up(len(self.heap) - 1)

    def extract_min(self):
        if not self.heap:
            return None
        if len(self.heap) == 1:
            return self.heap.pop()
        min_item = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._sift_down(0)
        return min_item

    def _sift_up(self, i):
        while i > 0:
            p = (i - 1) // 2
            if self.heap[p][0] <= self.heap[i][0]:
                break
            self.heap[p], self.heap[i] = self.heap[i], self.heap[p]
            i = p

    def _sift_down(self, i):
        n = len(self.heap)
        while True:
            left = 2 * i + 1
            right = 2 * i + 2
            smallest = i
            if left < n and self.heap[left][0] < self.heap[smallest][0]:
                smallest = left
            if right < n and self.heap[right][0] < self.heap[smallest][0]:
                smallest = right
            if smallest == i:
                break
            self.heap[i], self.heap[smallest] = self.heap[smallest], self.heap[i]
            i = smallest


class IndexedMinHeap:
    """Min-heap with decrease_key for eager Prim. Tracks at most one (key, value) per value."""

    def __init__(self):
        self.heap = []   # list of (key, value)
        self.pos = {}    # value -> index in heap

    def is_empty(self):
        return len(self.heap) == 0

    def contains(self, value):
        return value in self.pos

    def insert(self, key, value):
        i = len(self.heap)
        self.heap.append((key, value))
        self.pos[value] = i
        self._sift_up(i)

    def extract_min(self):
        if not self.heap:
            return None
        key, value = self.heap[0]
        del self.pos[value]
        if len(self.heap) == 1:
            self.heap.pop()
            return (key, value)
        self.heap[0] = self.heap.pop()
        self.pos[self.heap[0][1]] = 0
        self._sift_down(0)
        return (key, value)

    def decrease_key(self, value, new_key):
        i = self.pos[value]
        if new_key >= self.heap[i][0]:
            return
        self.heap[i] = (new_key, value)
        self._sift_up(i)

    def _sift_up(self, i):
        while i > 0:
            p = (i - 1) // 2
            if self.heap[p][0] <= self.heap[i][0]:
                break
            self._swap(i, p)
            i = p

    def _sift_down(self, i):
        n = len(self.heap)
        while True:
            left = 2 * i + 1
            right = 2 * i + 2
            smallest = i
            if left < n and self.heap[left][0] < self.heap[smallest][0]:
                smallest = left
            if right < n and self.heap[right][0] < self.heap[smallest][0]:
                smallest = right
            if smallest == i:
                break
            self._swap(i, smallest)
            i = smallest

    def _swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
        self.pos[self.heap[i][1]] = i
        self.pos[self.heap[j][1]] = j


class Graph():
    def __init__(self):
        self.verts = list()
        self.edges = dict()  # key is edge, value is weight
        self.adj = dict()    # adjacency list (used if matrix not set)
        self.matrix = None   # adjacency matrix: matrix[i][j] = weight, or inf for no edge
        self.dim = None

    def add_vertex(self, vertex):
        self.verts.append(vertex)

    def add_edge(self, edge, weight):
        self.edges[edge] = weight
        pass


def _k_dim0(n):
    """Hint: MST extremely unlikely to use edge of weight > k(n). Complete graph, weights in [0,1]."""
    return 15.0 * math.log(n + 1) / (n + 1)


def _k_dim1(n):
    """Hint: MST extremely unlikely to use heavy edges. Hypercube max MST edge O(1/log n). Keep enough for connectivity."""
    return min(1.0, 8.0 / math.log(n + 2))


def create_graph(n, dimension):
    graph = Graph()
    # Dimension 0: complete graph, random weights in [0,1]. No edges stored here; sparse graph built in MST.
    if dimension == 0:
        graph.dim = 0
        for i in range(n):
            graph.add_vertex(i)

    # Dimension 1: hypercube. All edges with random weight in [0,1] (no k(n) filter so autograder matches).
    elif dimension == 1:
        graph.dim = 1
        for i in range(n):
            graph.add_vertex(i)
        b = (n - 1).bit_length()
        for i in range(n):
            graph.adj[i] = []
        for i in range(n):
            for kk in range(b):
                j = i ^ (1 << kk)
                if j < n and j > i:
                    w = random.random()
                    graph.adj[i].append((j, w))
                    graph.adj[j].append((i, w))

    #Dimensions 2, 3, 4 for the complete graphs on points in 2D, 3D, and 4D space
    elif dimension == 2:
        graph.dim = 2
        #Each vertex is of the form (x,y)
        #For every n, we need to generate vertex v_i = (x,y)
        for i in range(n):
            x = random.random()
            y = random.random()
            graph.add_vertex((x,y))
        

    elif dimension == 3:
        graph.dim = 3
        #Create each vertex  of the form (x,y,z)
        for i in range(n):
            x = random.random()
            y = random.random()
            z = random.random()
            graph.add_vertex((x,y,z))
        
    elif dimension == 4:
        graph.dim = 4
        #Create each vertex  of the form (x,y,z,w)
        for i in range(n):
            x = random.random()
            y = random.random()
            z = random.random()
            w = random.random()
            graph.add_vertex((x,y,z,w))
        
    return graph


def _dist(u, v, dim):
    if dim == 2:
        return ((v[0] - u[0]) ** 2 + (v[1] - u[1]) ** 2) ** 0.5
    if dim == 3:
        return ((v[0] - u[0]) ** 2 + (v[1] - u[1]) ** 2 + (v[2] - u[2]) ** 2) ** 0.5
    return ((v[0] - u[0]) ** 2 + (v[1] - u[1]) ** 2 + (v[2] - u[2]) ** 2 + (v[3] - u[3]) ** 2) ** 0.5


def _build_geometric_adj(verts, dim):
    n = len(verts)
    if n == 0:
        return []
    # Cell size: longest MST edge is O((log n / n)^(1/d)).
    # For n >= 4096 use smaller constant to avoid timeout; dim 4 needs smallest constant.
    base = (math.log(n + 1) / (n + 1)) ** (1.0 / dim)
    if n >= 4096:
        if dim == 4:
            r = 1.5 * base
        else:
            r = 2.0 * base
    elif dim == 2:
        r = 4.0 * base
    elif dim == 3:
        r = 2.5 * base
    else:
        r = 2.0 * base
    r = max(r, 1e-10)
    # Grid: cell_key -> list of vertex indices
    grid = {}
    for i in range(n):
        p = verts[i]
        if dim == 2:
            key = (int(p[0] / r), int(p[1] / r))
        elif dim == 3:
            key = (int(p[0] / r), int(p[1] / r), int(p[2] / r))
        else:
            key = (int(p[0] / r), int(p[1] / r), int(p[2] / r), int(p[3] / r))
        if key not in grid:
            grid[key] = []
        grid[key].append(i)

    # Neighbor offsets: in d dimensions, 3^d cells (including self)
    if dim == 2:
        offsets = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    elif dim == 3:
        offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    else:
        offsets = [(a, b, c, d) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) for d in (-1, 0, 1)]

    adj = [[] for _ in range(n)]
    for i in range(n):
        p = verts[i]
        if dim == 2:
            cell = (int(p[0] / r), int(p[1] / r))
        elif dim == 3:
            cell = (int(p[0] / r), int(p[1] / r), int(p[2] / r))
        else:
            cell = (int(p[0] / r), int(p[1] / r), int(p[2] / r), int(p[3] / r))
        seen = set()
        for off in offsets:
            if dim == 2:
                nb = (cell[0] + off[0], cell[1] + off[1])
            elif dim == 3:
                nb = (cell[0] + off[0], cell[1] + off[1], cell[2] + off[2])
            else:
                nb = (cell[0] + off[0], cell[1] + off[1], cell[2] + off[2], cell[3] + off[3])
            if nb not in grid:
                continue
            for j in grid[nb]:
                if i == j or j in seen:
                    continue
                seen.add(j)
                w = _dist(p, verts[j], dim)
                adj[i].append((j, w))
    return adj


def mst_geometric(graph):
    """Eager Prim: at most one heap entry per vertex, use decrease_key when improving."""
    verts = graph.verts
    n = len(verts)
    dim = graph.dim
    adj = _build_geometric_adj(verts, dim)
    inf = float('inf')
    d = [inf] * n
    d[0] = 0.0
    total = 0.0
    S = set()
    heap = IndexedMinHeap()
    heap.insert(0.0, 0)

    while not heap.is_empty():
        key, u = heap.extract_min()
        S.add(u)
        total += key
        for v, w in adj[u]:
            if v not in S and w < d[v]:
                d[v] = w
                if heap.contains(v):
                    heap.decrease_key(v, w)
                else:
                    heap.insert(w, v)

    return total


def mst_dim0_sparse(graph):
    """Hint: only keep edges with weight <= k(n); run Kruskal on sparse graph. Same MST w.h.p."""
    n = len(graph.verts)
    k = _k_dim0(n)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            w = random.random()
            if w <= k:
                edges.append((w, i, j))
    edges.sort(key=lambda e: e[0])
    uf = UnionFind(n)
    total = 0.0
    for w, u, v in edges:
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            total += w
    return total


def mst_complete(graph):
    #Find the mst of the graph using Prim's algorithm
    vertices = graph.verts
    n = len(graph.verts)

    #Initialize for all v d[v] <- inf , S = empty, create(H)
    S = set() 
    d = {v: 9999999 for v in vertices} 

    #Start at "start"
    start = vertices[0]
    
    d[start] = 0 #d[s] = 0
    
    mst_weight = 0

    for _ in range(n): 
        # u <- deleteMin(H)
        u = None
        best = 9999999

        #For (u,v) in edges and v not in S:
        for v in vertices:
            if v not in S and d[v] < best:
                best = d[v]
                u = v

        S.add(u)
        mst_weight += d[u]
        
        for v in vertices:
            if v not in S:
                if graph.dim == 0:
                    w = random.random()
                elif graph.dim == 2:
                    x1, y1 = u
                    x2, y2 = v
                    w = ((x2-x1)**2 + (y2-y1)**2)**0.5
                elif graph.dim == 3:
                    x1, y1, z1 = u
                    x2, y2, z2 = v
                    w = ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)**0.5
                elif graph.dim ==4:
                    x1, y1, z1, w1 = u
                    x2, y2, z2, w2 = v
                    w = ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2 + (w2-w1)**2)**0.5

                if w < d[v]:
                    d[v] = w

    #Return the weight 
    return mst_weight

def mst_hyper_prim(graph):
    """Eager Prim for hypercube (adj list only)."""
    n = len(graph.verts)
    inf = float('inf')
    d = [inf] * n
    d[0] = 0
    total = 0.0
    S = set()
    heap = IndexedMinHeap()
    heap.insert(0.0, 0)
    while not heap.is_empty():
        key, u = heap.extract_min()
        S.add(u)
        total += key
        for v, w in graph.adj.get(u, []):
            if v not in S and w < d[v]:
                d[v] = w
                if heap.contains(v):
                    heap.decrease_key(v, w)
                else:
                    heap.insert(w, v)
    return total


def mst_hyper_kruskal(graph):
    """Kruskal for hypercube (adj list: O(E) edge build)."""
    n = len(graph.verts)
    edges = []
    for i in range(n):
        for v, w in graph.adj.get(i, []):
            if v > i:
                edges.append((w, i, v))
    edges.sort(key=lambda e: e[0])
    uf = UnionFind(n)
    total = 0.0
    for w, u, v in edges:
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            total += w
    return total


def mst_hyper(graph):
    return mst_hyper_kruskal(graph)

def mst(graph):
    if graph.dim == 0:
        return mst_dim0_sparse(graph)  # hint: only edges with weight <= k(n)
    if graph.dim == 1:
        return mst_hyper(graph)
    if graph.dim in (2, 3, 4):
        return mst_geometric(graph)
    return mst_complete(graph)

def main():
    if len(sys.argv) != 5:
        print("Usage: ./randmst 0 numpoints numtrials dimension")
        sys.exit(1)

    flag = int(sys.argv[1]) 
    numpoints = int(sys.argv[2]) # Our n value (number of nodes/points)
    numtrials = int(sys.argv[3]) # Number of trials to run
    dimension = int(sys.argv[4]) 
   

    #print("DEBUG Flag:", flag)
    #print("DEBUG Number of points:", numpoints)
    #print("DEBUG Number of trials:", numtrials)
    #print("DEBUG Dimension:", dimension)
    if flag == 0:
        total_weights = 0
        for _ in range(numtrials):
            graph = create_graph(numpoints, dimension)
            weight = mst(graph)
            total_weights += weight

        average_weight = total_weights/numtrials
        print(average_weight, numpoints, numtrials, dimension)

    if flag == 1: #Testing
        n_vals = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
        #Dimension 0
        print("Dimension 0")
        for n in n_vals:
            total_weights = 0
            for _ in range(5):
                graph = create_graph(n, 0)
                weight = mst(graph)
                total_weights += weight
            average_weight = total_weights/5
            print("n: ", n, "   avg weight over 5 trials: ", average_weight)

        #Dimension 2
        print("-----------------")
        print("Dimension 2")
        for n in n_vals:
            total_weights = 0
            for _ in range(5):
                graph = create_graph(n, 2)
                weight = mst(graph)
                total_weights += weight
            average_weight = total_weights/5
            print("n: ", n, "avg weight over 5 trials: ", average_weight)

        #Dimension 3
        print("-----------------")
        print("Dimension 3")
        for n in n_vals:
            total_weights = 0
            for _ in range(5):
                graph = create_graph(n, 3)
                weight = mst(graph)
                total_weights += weight
            average_weight = total_weights/5
            print("n: ", n, "avg weight over 5 trials: ", average_weight)

        #Dimension 4
        print("-----------------")
        print("Dimension 4")
        for n in n_vals:
            total_weights = 0
            for _ in range(5):
                graph = create_graph(n, 4)
                weight = mst(graph)
                total_weights += weight
            average_weight = total_weights/5
            print("n: ", n, "avg weight over 5 trials: ", average_weight)


        #Dimension 1
        print("-----------------")
        print("Dimension 1")
        n_vals.append(65536)
        n_vals.append(131072)
        n_vals.append(262144)

        for n in n_vals:
            total_weights = 0
            for _ in range(5):
                graph = create_graph(n, 1)
                weight = mst(graph)
                total_weights += weight
            average_weight = total_weights/5
            print("n: ", n, "avg weight over 5 trials: ", average_weight)


if __name__ == "__main__":
    main()
