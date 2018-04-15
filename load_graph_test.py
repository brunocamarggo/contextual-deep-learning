from igraph import *
if __name__ == '__main__':
    graph = Graph.Read_GraphMLz('graph.net')
    print(graph.vs['label'])
    # print(graph.get_adjacency())
   

    # adj_matrix = graph.get_adjacency()
    # print(adj_matrix)