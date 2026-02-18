#!/usr/bin/env python3

import sys
import random

class Graph():
    def __init__(self):
        self.verts = set()
        self.edges = dict() #key is edge, value is weight

    def add_vertex(self, vertex):
        self.verts.add(vertex)

    def add_edge(self, edge, weight):
        self.edges[edge] = weight
        pass


def create_graph(n, dimension):
    graph = Graph()
    #Dimension 0 for the complete graph with randomly distributed weights
    if dimension == 0:
        #Create vertices
        for i in range(n):
            graph.add_vertex(i)
        #Create edges with random weights
        for i in range(n):
            for j in range(i+1, n):
                weight = random.random()
                graph.add_edge((i,j), weight)
                

    #Dimension 1 for the hypercube graph with randomly distributed weights
    # n vertices numbered 0 through n-1, where (a,b) is an edge iff 
    # |a-b| = 2^i for some i, and the weight of each edge is a real 
    # number chosen uniformly at random on [0,1]
    elif dimension == 1:
        #Create vertices
        for i in range(n):
            graph.add_vertex(i)
        #Create edges with random weights
        for i in range(n):
            for j in range(i+1, n):
                if abs(i-j) & (abs(i-j) - 1) == 0: # Check if |a-b| is a power of 2
                    weight = random.random()
                    graph.add_edge((i,j), weight)

    #Dimensions 2, 3, 4 for the complete graphs on points in 2D, 3D, and 4D space
    elif dimension == 2:
        #Each vertex is of the form (x,y)
        #For every n, we need to generate vertex v_i = (x,y)
        for i in range(n):
            x = random.random()
            y = random.random()
            graph.add_vertex((x,y))
        #Create edges with weights as Euclidean distance
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = list(graph.verts)[i]
                x2, y2 = list(graph.verts)[j]
                weight = ((x2-x1)**2 + (y2-y1)**2)**0.5
                graph.add_edge((i,j), weight)

    elif dimension == 3:
        #Create each vertex  of the form (x,y,z)
        for i in range(n):
            x = random.random()
            y = random.random()
            z = random.random()
            graph.add_vertex((x,y,z))
        #Create edges with weights as Euclidean distance
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, z1 = list(graph.verts)[i]
                x2, y2, z2 = list(graph.verts)[j]
                weight = ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)**0.5
                graph.add_edge((i,j), weight)
    elif dimension == 4:
        #Create each vertex  of the form (x,y,z,w)
        for i in range(n):
            x = random.random()
            y = random.random()
            z = random.random()
            w = random.random()
            graph.add_vertex((x,y,z,w))
        #Create edges with weights as Euclidean distance
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, z1, w1 = list(graph.verts)[i]
                x2, y2, z2, w2 = list(graph.verts)[j]
                weight = ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2 + (w2-w1)**2)**0.5
                graph.add_edge((i,j), weight)
        
        

    print("DEBUG Created graph with Vertices: ", graph.verts, "and edges: ", graph.edges)
    return graph

def mst(graph):
    #Find the mst of the graph using Prim's algorithm
    vertices = list(graph.verts)
    edges = graph.edges

    #Initialize for all v d[v] <- inf , S = empty, create(H)
    mst = Graph() #S in Prim's
    heap = dict() #min-priority queue
    d = {v: 9999999 for v in vertices}
    prev = {v : None for v in vertices}

    #Start at "start"
    start = vertices[0]
    
    d[start] = 0 #d[s] = 0
    prev[start] = None #Prev[s] <- null
    heap[start] = 0 #Insert(H, s, 0)

    while len(heap) != 0: #while heap is not empty
        # u <- deleteMin(H)

        #S <- S U {u}
        
        #For (u,v) in edges and v not in S:
        #if d[v] > w((u,v))
        #d[v] = w((u,v)); Prev[v] = u; Insert(H,v,d[v])



    #Return the weight of that mst
    return None

def main():
    if len(sys.argv) != 5:
        print("Usage: ./randmst 0 numpoints numtrials dimension")
        sys.exit(1)

    flag = int(sys.argv[1]) 
    numpoints = int(sys.argv[2]) # Our n value (number of nodes/points)
    numtrials = int(sys.argv[3]) # Number of trials to run
    dimension = int(sys.argv[4]) 
   

    print("DEBUG Flag:", flag)
    print("DEBUG Number of points:", numpoints)
    print("DEBUG Number of trials:", numtrials)
    print("DEBUG Dimension:", dimension)

    total_weights = 0
    for i in range(numtrials):
        graph = create_graph(numpoints, dimension)
        weight = mst(graph)
        total_weights += weight

    average_weight = total_weights/numtrials
    print(average_weight, numpoints, numtrials, dimension)



if __name__ == "__main__":
    main()
