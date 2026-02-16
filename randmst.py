#!/usr/bin/env python3

import sys
import random

class Graph:
    def __init__(self):
        self.verts = []
        self.edges = []
    
    def add_vertex(self, v):
        self.verts.append(v)

    def add_edge(self, edge):
        self.edges.append(edge)

def create_graph(n, dimension):
    graph = Graph()
    #Dimension 0 for the complete graph with randomly distributed weights
    if dimension == 0:
        #Create vertices
        for i in range(n):
            graph.add_vertex(i)
            #ensure there is an edge from every vertex to every other vertex
            for j in graph.verts:
                graph.add_edge((i,j))

    #Dimension 1 for the hypercube graph with randomly distributed weights
    elif dimension == 1:
        #something

    #Dimensions 2, 3, 4 for the complete graphs on points in 2D, 3D, and 4D space
    elif dimension == 2:
        #something
    elif dimension == 3:
        #something
    elif dimension == 4:
        #something
    return graph

def mst(graph):
    #Find the mst of the graph


    #Return the weight of that mst
    return weight

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
