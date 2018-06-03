#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle
from collections import defaultdict
import itertools
from bcclasses import DetailedBoundBox
import igraph
import numpy as np
from bcloader import load_bottlenecks
from scipy import sparse 
from os.path import join
import config
from sklearn.model_selection import train_test_split

BOTTLENECK_PATH = config.paths['BOTTLENECK_PATH']
DATA_DIR = config.paths['DATA_DIR']
TESTING_PERCENTAGE = config.params['TESTING_PERCENTAGE']


def save_graph_as_dict(graph=None, file_name='ind.voc2012.graph'):
    """
    Saves a graph as a defaultdict(list) in the format {index: [index_of_neighbor_nodes]} as a pickle file.
    """
    adj_dict = defaultdict(list)
    for i, adjlist in enumerate(graph.get_adjlist()):
        adj_dict[i] = adjlist

    save_object(file_name=join(DATA_DIR, file_name), object_=adj_dict)


def global_class_to_one_hot(global_class):
    """
    Returns one-hot representation of a global class.
    """
    if global_class == 'Person':
        return [1, 0, 0, 0]
    elif global_class == 'Animal':
        return [0, 1, 0, 0]
    elif global_class == 'Vehicle':
        return [0, 0, 1, 0]
    elif global_class == 'Indoor':
        return [0, 0, 0, 1]
    else:
        raise Exception('The global class "' + global_class + '" cant be transform to one-hot format')


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
        raise Exception('The label "' + label + '" cant be transform to one-hot format')


def one_hot_to_label(one_hot=None):
    """
    Returns the label of a one-hot representation
    :param one_hot: a list of one-hot representation
    :return: The label of one-hot representation as string
    """
    if np.where(one_hot == 1)[0][0] == 0:
        return 'Person'
    elif np.where(one_hot == 1)[0][0] == 1:
        return 'Animal'
    elif np.where(one_hot == 1)[0][0] == 2:
        return 'Vehicle'
    elif np.where(one_hot == 1)[0][0] == 3:
        return 'Indoor'
    else:
        raise Exception('The one-hot representation "' + str(one_hot) + '" cant be transform to a label')


def get_one_hot_labels_list(labels_list=None):
    """
    Returns a list of all one-hot representation from a list with labels.
    """
    one_hot_labels = np.array([label_to_one_hot(label=label) for label in labels_list])
    return one_hot_labels


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
        raise Exception('The sub_class "' + sub_class + '" cant be transform to no one global class')

def get_global_class_MIT67(subclass=None):
    # auditorium, church_inside
    if subclass in ['bakery', 'grocerystore', 'clothingstore', 'deli',
               'laundromat', 'jewelleryshop', 'bookstore', 'videostore', 'florist', 'shoeshop', 'mall', 'toystore']:
        return 'Store'
    elif subclass in ['bedroom', 'nursery', 'closet', 'pantry', 'children_room', 'lobby', 'dining_room', 'corridor',
                    'livingroom', 'bathroom', 'kitchen', 'stairscase', 'winecellar', 'garage']:
        return 'Home'
    elif subclass in ['auditorium', 'prisoncell', 'church_inside', 'library', 'cloister', 'church', 'waitingroom', 'museum', 'elevator', 'poolinside',
                      'inside_bus', 'inside_subway', 'subway', 'locker_room', 'trainstation', 'airport_inside']:
        return 'Public spaces'
    elif subclass in ['buffet', 'fastfood_restaurant', 'concert_hall', 'restaurant', 'bar', 'movietheater', 'gameroom', 'casino',
                      'bowling', 'gym', 'hairsalon']:
        return 'Leisure'
    elif subclass in ['hospitalroom', 'kindergarden', 'restaurant_kitchen', 'artstudio', 'classroom', 'laboratorywet',
                      'studiomusic', 'operating_room', 'office', 'computerroom', 'warehouse', 'greenhouse',
                      'dentaloffice', 'tv_studio', 'meeting_room']:
        return 'Working place'
    else:
        raise Exception('The subclass "' + subclass + '" cant be transform to no one global class (MIT67).')


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
    print("Getting graph's vertex and edges. Please wait...")
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
    edges_list = remove_duplicate_edges(edges_list)
    print('\tNUMBER OF VERTEXES: {}'.format(len(bound_box_list)))
    print('\tNUMBER OF EDGES: {}'.format(len(edges_list)))
    print('\t[DONE]')
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


def save_object(file_name=None, object_=None):
    """
    Saves an object as pickle format
    :param file_name: name to save the file
    :param object_: the object to be saved
    :return: None
    """
    with open(file_name, 'wb') as output:
        pickle.dump(object_, output, protocol=2)
        output.close()


def load_bottleneck_values(bottlenecskpath=BOTTLENECK_PATH, bottleneck_file=None):
    """
    Reads a bottleneck file and returns the float values gifts inside it.
    :param bottleneck_file:
    :return: A list with the float values of a bottleneck file.
    """
    values = []
    with open(join(bottlenecskpath, bottleneck_file), 'r') as input_:
        for line in input_.readlines():
            aux = [float(value) for value in line.split(",")]
            values.append(aux)
    input_.close()
    return values


def compare_labels(bottleneck_file=None, other=None):
    """
    Compare if two classes are equals.
    :param bottleneck_file: bottleneck file name
    :param other: Another class name to compare
    :return: None
    """
    bottleneck_name = bottleneck_file.split('_')[3].replace('.jpg', '').replace('.txt', '')
    global_class = get_global_class(sub_class=bottleneck_name)
    if global_class != other:
        raise Exception('Feature representation not matching with one-hot representation')


def verify_labels_order(x_test_indices=None, graph_labels=None, y_test_labels=None):
    """
    Verifies if the labels of graph, x_test_indices and y_test_labels are the same.
    """
    test = True
    for k, i in enumerate(x_test_indices):
        if get_global_class(sub_class=graph_labels[i]) != y_test_labels[k]:
            test = False
            break
    if not test:
        raise Exception('Label from graph does not match with testing data')


def generate_inputs_files(dataset_name='voc2012', graph=None, one_hot_labels_list=None):
    """
    Produces all input files required to use Thomas N. Kipf and Max Welling implementation of Graph Convolutional

    Networks. Look more at:
        https://github.com/tkipf/gcn

    Thomas N. Kipf and Max Welling made use of data split provided by (Zhilin Yang, William W. Cohen,
    Ruslan Salakhutdinov, Revisiting Semi-Supervised Learning with Graph Embeddings, ICML 2016). Look more at:
        https://github.com/kimiyoung/planetoid

    list of one-hot training instances = y
    list of one-hot testing list instances = ty
    list of features training instances = x
    list of features testing instances = tx
    list of the indices of test instances in graph = test.index
    Graph, a dict in the format {index: [index_of_neighbor_nodes]}

    :param dataset_name: dataset name
    :param graph: an igraph Object
    :param one_hot_labels_list: list of all one-hot labels
    :return: None
    """
    graph.write_graphmlz(join(DATA_DIR, 'graph.net'))
    indices = [i for i in range(len(one_hot_labels_list))]

    y = []
    for one_hot in one_hot_labels_list:
        y.append(one_hot_to_label(one_hot=one_hot))

    bottlenecks = load_bottlenecks()
    X = []
    for k, i in enumerate(indices):
        compare_labels(bottleneck_file=bottlenecks[i], other=y[k])
        bottlenecks_values = load_bottleneck_values(bottleneck_file=bottlenecks[i])
        for values in bottlenecks_values:
            X.append(values)

    allx, tx, ally, ty, allx_indices, X_test_indices = train_test_split(X, y, indices, stratify=y,
                                                                        test_size=TESTING_PERCENTAGE)

    ally = [global_class_to_one_hot(global_class=ally_) for ally_ in ally]
    labels = graph.vs['label']
    verify_labels_order(graph_labels=labels, y_test_labels=ty, x_test_indices=X_test_indices)
    ty = [global_class_to_one_hot(global_class=ty_) for ty_ in ty]

    allx_indices = [i for i in range(len(allx_indices))]
    # x e y are samples with labels from training data
    #  x_ e y_ are samples with no labels from training data
    x_, x, y_, y, x_train_indices, x_test_indices = train_test_split(allx, ally, allx_indices, stratify=ally,
                                                                     test_size=TESTING_PERCENTAGE)
    x = sparse.csr_matrix(x)
    tx = sparse.csr_matrix(tx)
    allx = sparse.csr_matrix(allx)
    y = np.array(y)
    ty = np.array(ty)
    ally = np.array(ally)

    save_object(file_name=join(DATA_DIR, 'ind.' + dataset_name + '.x'), object_=x)
    save_object(file_name=join(DATA_DIR, 'ind.' + dataset_name + '.tx'), object_=tx)
    save_object(file_name=join(DATA_DIR, 'ind.' + dataset_name + '.allx'), object_=allx)
    save_object(file_name=join(DATA_DIR, 'ind.' + dataset_name + '.y'), object_=y)
    save_object(file_name=join(DATA_DIR, 'ind.' + dataset_name + '.ty'), object_=ty)
    save_object(file_name=join(DATA_DIR, 'ind.' + dataset_name + '.ally'), object_=ally)
    save_object(file_name=join(DATA_DIR, 'ind.' + dataset_name + '.test.index'), object_=X_test_indices)
    save_graph_as_dict(graph=graph)



