import config
from igraph import Graph
from os import listdir
from evaluate_MIT67_dataset import evaluate
from itertools import  permutations
from bcutils import save_graph_as_dict, plot_graph
import sys
import mit67


IMG_DIR = config.paths['IMG_DIR_MIT67']


if __name__ == '__main__':
    images, bottlenecks, dict_ = evaluate()
    # images = array com todos as imagens, ou seja, todos os nos do grafo
    # edges = lista a ser montada com todas as licoes
    edges = []
    step = 0
    start = 0
    for key in sorted(dict_):
        for i in range(start, start + dict_[key]):
            for j in range(start, step):
                edge = (i, j)
                edges.append(edge)
            step += 1
        start += dict_[key]
    print(len(edges))

    LABELS = [img.split('*')[0] for img in images]
    NUM_VERTICES = len(images)

    del images
    graph = Graph()
    graph.add_vertices(NUM_VERTICES)
    graph.add_edges(edges)
    graph.vs['label'] = LABELS
    print('Images: {}, V: {}, E: {}'.format(NUM_VERTICES, NUM_VERTICES, len(edges)))
    del edges
    print('\t[DONE]')

    #plot_graph(graph=graph)
    save_graph_as_dict(graph=graph, file_name='ind.mit67.graph')
    one_hot_list = mit67.get_one_hot_labels_list(labels_list=LABELS)
    mit67.generate_inputs_files(graph=graph, one_hot_labels_list=one_hot_list, bottlenecks=bottlenecks)
   # plot_graph(graph)

    # permus = remove_duplicate_edges(vertex_list=permus)
    # print(len(permus))