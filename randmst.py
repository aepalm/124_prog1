#!/usr/bin/env python3

import sys
import random

class Graph():
    def __init__(self):
        self.verts = list()
        self.edges = dict() #key is edge, value is weight

    def add_vertex(self, vertex):
        self.verts.append(vertex)

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
            u = graph.verts[i]
            for j in range(i+1, n):
                v = graph.verts[j]
                x1, y1 = u
                x2, y2 = v
                weight = ((x2-x1)**2 + (y2-y1)**2)**0.5
                graph.add_edge((u,v), weight)

    elif dimension == 3:
        #Create each vertex  of the form (x,y,z)
        for i in range(n):
            x = random.random()
            y = random.random()
            z = random.random()
            graph.add_vertex((x,y,z))
        #Create edges with weights as Euclidean distance
        for i in range(n):
            u = graph.verts[i]
            for j in range(i+1, n):
                v = graph.verts[j]
                x1, y1, z1 = u
                x2, y2, z2 = v
                weight = ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)**0.5
                graph.add_edge((u,v), weight)
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
            u = graph.verts[i]
            for j in range(i+1, n):
                v = graph.verts[j]
                x1, y1, z1, w1 = u
                x2, y2, z2, w2 = v
                weight = ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2 + (w2-w1)**2)**0.5
                graph.add_edge((u,v), weight)
        
        

    #print("DEBUG Created graph with Vertices: ", graph.verts, "and edges: ", graph.edges)
    return graph

def mst(graph,n): #n is the number of vertices
    #Find the mst of the graph using Prim's algorithm
    vertices = graph.verts
    edges = graph.edges

    #Initialize for all v d[v] <- inf , S = empty, create(H)
    S = set() 
    d = {v: 9999999 for v in vertices} 
    prev = {v : None for v in vertices}

    #Start at "start"
    start = vertices[0]
    
    d[start] = 0 #d[s] = 0
    prev[start] = None #Prev[s] <- null
    
    mst_weight = 0

    while len(S) < n: 
        # u <- deleteMin(H)
        u = None
        min = 9999999

        #For (u,v) in edges and v not in S:
        for v in vertices:
            if v not in S and d[v] < min:
                min = d[v]
                u = v

        S.add(u)
        mst_weight += d[u]
        #if d[v] > w((u,v))
        #d[v] = w((u,v)); Prev[v] = u;
        for (a,b), weight in edges.items():
            if a == u and b not in S:
                if weight < d[b]:
                    d[b] = weight
                    prev[b] = u
            elif b == u and a not in S:
                if weight < d[a]:
                    d[a] = weight
                    prev[a] = u

    #Return the weight 
    return mst_weight

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
        for i in range(numtrials):
            graph = create_graph(numpoints, dimension)
            weight = mst(graph,numpoints)
            total_weights += weight

        average_weight = total_weights/numtrials
        print(average_weight, numpoints, numtrials, dimension)

    if flag == 1: #Testing
        n_vals = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
        #Dimension 0
        print("Dimension 0")
        for n in n_vals:
            total_weights = 0
            for j in range(5):
                graph = create_graph(n, 0)
                weight = mst(graph,n)
                total_weights += weight
            average_weight = total_weights/5
            print("n: ", n, "avg weight over 5 trials: ", average_weight)

        #Dimension 2
        print("-----------------")
        print("Dimension 2")
        for n in n_vals:
            total_weights = 0
            for j in range(5):
                graph = create_graph(n, 2)
                weight = mst(graph,n)
                total_weights += weight
            average_weight = total_weights/5
            print("n: ", n, "avg weight over 5 trials: ", average_weight)

        #Dimension 3
        print("-----------------")
        print("Dimension 3")
        for n in n_vals:
            total_weights = 0
            for j in range(5):
                graph = create_graph(n, 3)
                weight = mst(graph,n)
                total_weights += weight
            average_weight = total_weights/5
            print("n: ", n, "avg weight over 5 trials: ", average_weight)

        #Dimension 4
        print("-----------------")
        print("Dimension 4")
        for n in n_vals:
            total_weights = 0
            for j in range(5):
                graph = create_graph(n, 4)
                weight = mst(graph,n)
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
            for j in range(5):
                graph = create_graph(n, 1)
                weight = mst(graph,n)
                total_weights += weight
            average_weight = total_weights/5
            print("n: ", n, "avg weight over 5 trials: ", average_weight)






if __name__ == "__main__":
    main()
