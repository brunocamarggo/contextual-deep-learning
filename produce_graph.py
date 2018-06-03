#!/usr/bin/env python
# -*- coding: utf-8 -*-
from igraph import *
from bcutils import (
    get_one_hot_labels_list,
    get_all_bound_boxes,
    generate_inputs_files
    )
from bcloader import load_xmls
import config

XML_DIR = config.paths['XML_DIR']
DATA_DIR = config.paths['DATA_DIR']

if __name__ == '__main__':
    all_xmls = load_xmls(path=XML_DIR)
    # 1500 - best
    all_bndboxes, edges_list = get_all_bound_boxes(all_xmls[:1500])

    print('Building the graph. Please wait...')
    NUMBER_OF_VERTICES = len(all_bndboxes)

    LABELS = [bndbox.sub_class for bndbox in all_bndboxes]    
    graph = Graph()
    graph.add_vertices(NUMBER_OF_VERTICES)
    graph.add_edges(edges_list)
    graph.vs['label'] = LABELS
    print('\t[DONE]')

    # saving files

    one_hot_labels = get_one_hot_labels_list(labels_list=LABELS)
    generate_inputs_files(graph=graph, one_hot_labels_list=one_hot_labels)

    
  



