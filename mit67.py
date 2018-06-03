import pickle
from collections import defaultdict
import itertools
from bcclasses import DetailedBoundBox
import igraph
from bcutils import save_object
import numpy as np
from bcloader import load_bottlenecks
from scipy import sparse
from os.path import join
import config
from sklearn.model_selection import train_test_split
from bcutils import load_bottleneck_values, save_graph_as_dict

BOTTLENECK_PATH = config.paths['BOTTLENECK_PATH_MIT67']
DATA_DIR = config.paths['DATA_DIR']
TESTING_PERCENTAGE = config.params['TESTING_PERCENTAGE']


def label_to_one_hot(label=None):
    if label == 'Store':
        return [1, 0, 0, 0, 0]
    elif label == 'Home':
        return [0, 1, 0, 0, 0]
    elif label == 'Public spaces':
        return [0, 0, 1, 0, 0]
    elif label == 'Leisure':
        return [0, 0, 0, 1, 0]
    elif label == 'Working place':
        return [0, 0, 0, 0, 1]
    else:
        raise Exception('The label "' + label + '" cant be transform to one-hot format')


def get_one_hot_labels_list(labels_list=None):
    """
    Returns a list of all one-hot representation from a list with labels.
    """
    one_hot_labels = np.array([label_to_one_hot(label=label) for label in labels_list])
    return one_hot_labels


def one_hot_to_label(one_hot=None):
    """
    Returns the label of a one-hot representation
    :param one_hot: a list of one-hot representation
    :return: The label of one-hot representation as string
    """
    if np.where(one_hot == 1)[0][0] == 0:
        return 'Store'
    elif np.where(one_hot == 1)[0][0] == 1:
        return 'Home'
    elif np.where(one_hot == 1)[0][0] == 2:
        return 'Public spaces'
    elif np.where(one_hot == 1)[0][0] == 3:
        return 'Leisure'
    elif np.where(one_hot == 1)[0][0] == 4:
        return 'Working place'
    else:
        raise Exception('The one-hot representation "' + str(one_hot) + '" cant be transform to a label')


def global_class_to_one_hot(global_class):
    """
    Returns one-hot representation of a global class.
    """
    if global_class == 'Store':
        return [1, 0, 0, 0, 0]
    elif global_class == 'Home':
        return [0, 1, 0, 0, 0]
    elif global_class == 'Public spaces':
        return [0, 0, 1, 0, 0]
    elif global_class == 'Leisure':
        return [0, 0, 0, 1, 0]
    elif global_class == 'Working place':
        return [0, 0, 0, 0, 1]
    else:
        raise Exception('The global class "' + global_class + '" cant be transform to one-hot format')


def verify_labels_order(x_test_indices=None, graph_labels=None, y_test_labels=None):
    """
    Verifies if the labels of graph, x_test_indices and y_test_labels are the same.
    """
    test = True
    for k, i in enumerate(x_test_indices):
        if graph_labels[i] != y_test_labels[k]:
            test = False
            break
    if not test:
        raise Exception('Label from graph does not match with testing data')


def generate_inputs_files(dataset_name='mit67', graph=None, one_hot_labels_list=None, bottlenecks=None):
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

    X = []
    for k, i in enumerate(indices):
        # compare_labels(bottleneck_file=bottlenecks[i], other=y[k])
        # Example: 'Home*winecellar*wine_storage_42_02_altavista.jpg.txt
        if not y[k] == bottlenecks[i].split('*')[0]:
            raise Exception('Feature representation not matching with one-hot representation')

        filename = bottlenecks[i].split('*')[1] + '*' +bottlenecks[i].split('*')[2]
        bottlenecks_values = load_bottleneck_values(bottlenecskpath=BOTTLENECK_PATH,
                                                    bottleneck_file=filename)
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

