#!/usr/bin/env python3


#Usage: ./randmst 0 numpoints numtrials dimension
#To get values for table/graph, use ./randmst 1 0 0 0
import sys
import random
import math

class Graph():
    def __init__(self):
        self.verts = list()
        self.edges = dict() #key is edge, value is weight
        self.adj = None #adjacency list
        self.dim = None

    def add_vertex(self, vertex):
        self.verts.append(vertex)

    def add_edge(self, edge, weight):
        self.edges[edge] = weight
        pass


def create_graph(n, dimension):
    graph = Graph()
    #Dimension 0 for the complete graph with randomly distributed weights
    if dimension == 0:
        graph.dim = 0
        #Create vertices
        for i in range(n):
            graph.add_vertex(i)
                

    #Dimension 1 for the hypercube graph with randomly distributed weights
    elif dimension == 1:
        graph.dim = 1
        #Create vertices
        for i in range(n):
            graph.add_vertex(i)
        #Create edges with random weights
        b = (n-1).bit_length()
        for i in range(n):
            for k in range(b):
                j = i ^ (1 << k)
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
        min = 9999999

        #For (u,v) in edges and v not in S:
        for v in vertices:
            if v not in S and d[v] < min:
                min = d[v]
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

def mst_hyper(graph):
    n = len(graph.adj)
    S = set()
    d = [999999] * n

    d[0] = 0
    total = 0


    for _ in range(n):
        u = None
        min = 999999
        for i in range(n):
            if i not in S and d[i] < min:
                min = d[i]
                u = i
        
        S.append(u)
        total += min

        for v, w in graph.adj[u]:
            if v not in S and w < d[v]:
                d[v] = w

    return total

def mst(graph):
    if graph.dim == 1:
        return mst_hyper(graph)
    else:
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
