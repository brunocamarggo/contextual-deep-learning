from os import listdir
import numpy as np
from bcutils import save_object
from scipy import sparse
from os.path import join
import config
from sklearn.model_selection import train_test_split
from bcutils import load_bottleneck_values, save_graph_as_dict

DATASET_DIR = config.paths['IMG_DIR_SYNTHETIC']
BOTTLENECKS_PATH = config.paths['BOTTLENECK_PATH_SYNTHETIC']
DATA_DIR = config.paths['DATA_DIR']
TESTING_PERCENTAGE = config.params['TESTING_PERCENTAGE']


def evaluate():
    images = [img for img in sorted(listdir(DATASET_DIR))]
    bottlenecks = [b for b in sorted(listdir(BOTTLENECKS_PATH))]
    # for i, img in enumerate(images):
    #     if not bottlenecks[i].replace('.txt', '') == img:
    #         raise Exception('Err')

    dataset_dict = {}
    for i, img in enumerate(images):
        class_ = img.split('_')[0]
        images[i] = class_ + '*' + img
        # bottlenecks[i] = class_ + '*' + bottlenecks[i]
        if class_ not in dataset_dict:
            dataset_dict[class_] = 1
        else:
            dataset_dict[class_] += 1

    print(dataset_dict)
    return sorted(images), sorted(bottlenecks), dataset_dict


def label_to_one_hot(label=None):
    if label == 'circle':
        return [1, 0, 0]
    elif label == 'rectangle':
        return [0, 1, 0]
    elif label == 'triangle':
        return [0, 0, 1]
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
        return 'circle'
    elif np.where(one_hot == 1)[0][0] == 1:
        return 'rectangle'
    elif np.where(one_hot == 1)[0][0] == 2:
        return 'triangle'
    else:
        raise Exception('The one-hot representation "' + str(one_hot) + '" cant be transform to a label')


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



def generate_inputs_files(dataset_name='synthetic', graph=None, one_hot_labels_list=None, bottlenecks=None):
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
        if not y[k] == bottlenecks[i].split('_')[0]:
            raise Exception('Feature representation not matching with one-hot representation')

        print(bottlenecks[i])
        filename = bottlenecks[i]
        bottlenecks_values = load_bottleneck_values(bottlenecskpath=BOTTLENECKS_PATH,
                                                    bottleneck_file=filename)
        for values in bottlenecks_values:
            X.append(values)

    allx, tx, ally, ty, allx_indices, X_test_indices = train_test_split(X, y, indices, stratify=y,
                                                                        test_size=TESTING_PERCENTAGE)

    ally = [label_to_one_hot(label=ally_) for ally_ in ally]
    labels = graph.vs['label']
    verify_labels_order(graph_labels=labels, y_test_labels=ty, x_test_indices=X_test_indices)
    ty = [label_to_one_hot(label=ty_) for ty_ in ty]

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


