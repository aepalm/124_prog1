#!/usr/bin/env python3

import sys
import random

def create_graph(n, dimension):
    graph = Graph()
    #Dimension 0 for the complete graph with randomly distributed weights
    if dimension == 0:
        #Create vertices
        for i in range(n):
            
                

    #Dimension 1 for the hypercube graph with randomly distributed weights
    elif dimension == 1:
        #something

    #Dimensions 2, 3, 4 for the complete graphs on points in 2D, 3D, and 4D space
    elif dimension == 2:
        #Each vertex is of the form (x,y)
        #For every n, we need to generate vertex v_i = (x,y)

        #Each edge weighting is the Euclidean distance 
    elif dimension == 3:
        #Each vertex is of the form (x,y,z)
        #Each edge weighting is the Euclidean distance
    elif dimension == 4:
        #Each vertex is of the form (x,y,z,w)
        #Each edge weighting is the Euclidean distance
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
