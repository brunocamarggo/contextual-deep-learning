import pickle
from collections import defaultdict
import itertools
from bcclasses import DetailedBoundBox
import igraph
import numpy as np
from bcloader import load_bottlenecks
from scipy import sparse 
from os.path import isfile, join
import config

BOTTLENECK_PATH = config.paths['BOTTLENECK_PATH']
DATA_DIR = config.paths['DATA_DIR']


def save_graph_as_dict(graph=None, file_name=join(DATA_DIR, 'graph.pkl')):
    """
    Saves a graph as a defaultdict(list) in the format {index: [index_of_neighbor_nodes]} as a pickle file.
    """
    adj_dict =  defaultdict(list)
    for i, adjlist in enumerate(graph.get_adjlist()):
        adj_dict[i] = adjlist
    
    with open(file_name, 'wb') as output:
        pickle.dump(adj_dict, output, pickle.HIGHEST_PROTOCOL)


def load_object(object_path=None):
    """
    Loads an objects in pickle format.
    """
    with open(object_path, 'rb') as input_:
        graph_dict = pickle.load(input_)
        return graph_dict


def label_to_one_hot(label=None):
    """
        Returns a label in format one-hot.
    """
    if get_global_class(label) == 'Person':
        return [1, 0, 0, 0]
    elif get_global_class(label) == 'Animal':
        return [0, 1, 0, 0]
    elif get_global_class(label) == 'Vehicle':
        return [0, 0, 1, 0]
    elif get_global_class(label) == 'Indoor':
        return [0, 0, 0, 1]
    else:
        raise Exception('The label "'+ label +'" cant be transform to one-hot format')


def get_one_hot_labels_list(labels_list=None):
    """
    Returns a list of all one-hot represetation from a list with labels.
    """
    one_hot_labels = np.array([label_to_one_hot(label=label) for label in labels_list])
    return one_hot_labels


def save_one_hot_labels_list(one_hot_labels_list=None, file_name=join(DATA_DIR, "one_hot_labels_list.pkl")):
    with open(file_name, 'wb') as output:
        pickle.dump(one_hot_labels_list, output, pickle.HIGHEST_PROTOCOL)


def get_global_class(sub_class=None):
    """ 
        Just a simple method to return a global class of a sub_class known in advance.
    """
    if sub_class in ['person']:
        return 'Person'
    elif sub_class in ['bird', 'cat', 'cow', 'dog', 'horse', 'sheep']:
        return 'Animal'
    elif sub_class in ['aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train']:
        return 'Vehicle'
    elif sub_class in ['bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']:
        return 'Indoor'
    else:
        raise Exception('The sub_class "'+ sub_class +'" cant be transform to no one global class')


def remove_duplicate_edges(vertex_list):
    """
        Remove all duplicate duples present in the vertex list. For example:
        vertex list = [(0, 1), (1, 0)]. So, the return will be [(0, 1)].
    """
    temp = []
    for a,b in vertex_list :
        if (a,b) not in temp and (b,a) not in temp: #to check for the duplicate tuples
            temp.append((a,b))
    vertex_list = temp * 1 #copy temp to d
    return vertex_list


def get_all_bound_boxes(annotation_list=None):
    """
        Returns all bound boxes (here treated like DetailedBoundBox) present in an annotation list, that is, a list
        with xml files builded in advance.
        This method also provide a list containing the edges with a bound box to others bound box present in the same
        image.
    """
    print("Getting graph's vertex and edges. Please wait")
    bound_box_list = []
    edges_list = []
    step = 0
    for i, annotation in enumerate(annotation_list):
        if len(annotation.objects) >= 2:
            iterable = list(range(step, len(annotation.objects)+step))
            step += len(annotation.objects)
            edges_list += itertools.permutations(iterable, 2)
            for obj in annotation.objects:
                detailedBoudBox = DetailedBoundBox(
                    get_global_class(sub_class=obj.name),
                    obj.name,
                    obj.get_bndbox(),
                    annotation.filename
                )
                bound_box_list.append(detailedBoudBox)
        if i % 100 == 0:
            print('\t{}/{} xmls processed'.format(i, len(annotation_list)))
    edges_list = remove_duplicate_edges(edges_list)
    print('\t [DONE]')
    return bound_box_list, edges_list


def plot_graph(graph=None):
    """
    Plot a graph with some customization. Dont waste your time trying to plot huge graphs hahaha.
    """
    layout = graph.layout("kk")
    visual_style = {}
    visual_style["layout"] = layout
    visual_style["vertex_label_size"] = 10
    visual_style["bbox"] = (800, 600)
    visual_style["margin"] = 50
    igraph.plot(graph,**visual_style)


def save_feature_list(size=0, file_name=join(DATA_DIR, 'features.pkl')):
    """
    Saves all freature array as a sparse matrix.
    """
    bottlenecks = load_bottlenecks()
    out_list = []
    for i in range(size):
        with open(join(BOTTLENECK_PATH, bottlenecks[i]), 'r') as input_:
            for line in input_.readlines():
                aux = [float(value) for value in line.split(",")]
                out_list.append(aux)
    
    out_list = sparse.csr_matrix(out_list) 
          
    with open(file_name, 'wb') as output:
        pickle.dump(out_list, output, pickle.HIGHEST_PROTOCOL)