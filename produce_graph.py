from igraph import *
from bcutils import (
    save_graph_as_dict, 
    get_one_hot_labels_list,
    get_all_bound_boxes,
    save_one_hot_labels_list,
    save_feature_list
    )
from bcloader import load_xmls
from os.path import join
import config

XML_DIR = config.paths['XML_DIR']
DATA_DIR = config.paths['DATA_DIR']

if __name__ == '__main__':
    all_xmls = load_xmls(path=XML_DIR)  
    all_bndboxes, edges_list = get_all_bound_boxes(all_xmls[:1500])

    print('Building the graph. Please wait...')
    NUMBER_OF_VERTICES = len(all_bndboxes)
    LABELS = [bndbox.sub_class for bndbox in all_bndboxes]    
    graph = Graph()
    graph.add_vertices(NUMBER_OF_VERTICES)
    graph.add_edges(edges_list)
    graph.vs['label'] = LABELS
    print('\t[DONE]')

    one_hot_labels = get_one_hot_labels_list(labels_list=LABELS)
    # saving files 
    save_one_hot_labels_list(one_hot_labels_list=one_hot_labels)
    save_graph_as_dict(graph=graph)
    save_feature_list(size=NUMBER_OF_VERTICES)
    graph.write_graphmlz(join(DATA_DIR, 'graph.net'))

    
  



