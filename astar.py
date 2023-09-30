# @ Viranjan Bhattacharyya
# An implementation of the A* algorithm
# Finding the shortest path on a grid/maze
# Pseudo code: https://isaaccomputerscience.org/concepts/dsa_search_a_star?examBoard=all&stage=all

import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, coord=None, parent=None):
        self.coord = coord
        self.parent = parent
        self.f = 100
        self.g = 100

    # defining comparision between nodes    
    def __eq__(self, other):
        return self.coord == other.coord
    
    def __lt__(self, other):
        return self.f < other.f
    
    def __gt__(self, other):
        return self.f > other.f

class AStar:
    def __init__(self, Grid, goal_coord, move_diagonal=True):
        self.grid_ = Grid
        self.goal_node = Node(goal_coord)

        if move_diagonal:
            self.neighbor_vectors = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]
        else:
            self.neighbor_vectors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        self.visited = []
        self.unvisited = []
        # add all nodes to unvisited with default high cost and null parent
        for m in range(self.grid_.shape[0]):
            for n in range(self.grid_.shape[1]):
                self.unvisited.append(Node((m, n)))

        self.max_iter_ = self.grid_.shape[0] * self.grid_.shape[1]
        self.path_= []

    def heuristic(self, node):
        """ Euclidean Distance Heuristic """
        h = np.sqrt((node.coord[0] - self.goal_node.coord[0]) ** 2 + (node.coord[1] - self.goal_node.coord[1]) ** 2)
        return h
    
    def search(self, start_coord):
        start_node = Node(start_coord)
        start_node = self.unvisited[self.unvisited.index(start_node)]
        start_node.g = 0
        start_node.f = self.heuristic(start_node)

        iter = 0
        while len(self.unvisited) > 0:
            iter += 1
            if iter > self.max_iter_:
                print("goal not found in time")
                break

            current_node = self.unvisited[self.unvisited.index(min(self.unvisited))]

            if current_node == self.goal_node:
                self.visited.append(current_node)
                self.unvisited.remove(current_node)
                break

            else:
                nbr_list = []
                for v in self.neighbor_vectors:
                    nbr_coord = (current_node.coord[0] + v[0], current_node.coord[1] + v[1])
                    nbr_node = Node(nbr_coord)
                    if nbr_node in self.unvisited:
                        if self.grid_[nbr_coord[0]][nbr_coord[1]] != 1: # Walls are marked with 1
                            nbr_list.append(self.unvisited[self.unvisited.index(nbr_node)])
                        else:
                            continue
                    else:
                        continue

                for nbr in nbr_list:
                    if not nbr in self.visited:
                        g_new = current_node.g + self.grid_[nbr.coord[0]][nbr.coord[1]]
                        if g_new < nbr.g:
                            nbr.g = g_new
                            nbr.f = nbr.g + self.heuristic(nbr)
                            nbr.parent = current_node
                
                self.visited.append(current_node)
                self.unvisited.remove(current_node)
                print(iter)

        self.return_path(current_node)

    def return_path(self, current_node):
        path = []
        current = current_node
        while current is not None:
            path.append(current.coord)
            current = current.parent
        self.path_ = path[::-1]  # Return reversed path


if __name__ == '__main__':    
    # Example Test Grid: https://www.analytics-link.com/post/2018/09/14/applying-the-a-path-finding-algorithm-in-python-part-1-2d-square-Grid
    Grid = np.array([
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    start = (0, 0)
    goal = (0, 19)

    astar = AStar(Grid, goal)
    astar.search(start)

    path = astar.path_
    x = []
    y = []
    for i in (range(len(path))):
        x.append(path[i][0])
        y.append(path[i][1])
    # plot map and path
    fig, ax = plt.subplots(figsize=(12,12))
    ax.imshow(Grid, cmap=plt.cm.Dark2)
    ax.scatter(start[1],start[0], marker = "*", color = "yellow", s = 200)
    ax.scatter(goal[1],goal[0], marker = "*", color = "red", s = 200)
    ax.plot(y,x, color = "black")

    plt.show()