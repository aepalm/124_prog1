#!/usr/bin/env python3

import sys

def main():
    if len(sys.argv) != 5:
        print("Usage: ./randmst 0 numpoints numtrials dimension")
        sys.exit(1)

    flag = int(sys.argv[1])
    numpoints = int(sys.argv[2])
    numtrials = int(sys.argv[3])
    dimension = int(sys.argv[4])

    print("Flag:", flag)
    print("Number of points:", numpoints)
    print("Number of trials:", numtrials)
    print("Dimension:", dimension)

    # Your MST logic here


if __name__ == "__main__":
    main()
