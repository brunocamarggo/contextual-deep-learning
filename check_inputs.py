import pickle
from scipy import sparse

import bcutils

if __name__ == '__main__':
    """
    
    https://github.com/kimiyoung/planetoid

    Transductive learning
    The input to the transductive model contains:

    x, the feature vectors of the training instances,
    y, the one-hot labels of the training instances,
    graph, a dict in the format {index: [index_of_neighbor_nodes]}, where the neighbor nodes are organized as a list. 
    The current version only supports binary graphs.
    Let L be the number of training instances. The indices in graph from 0 to L - 1 must correspond to the training 
    instances, with the same order as in x.
    """
    # x
    object_ = bcutils.load_object(object_path='/home/bruno/Dropbox/utfpr/tcc2/features.pkl')
    print(type(object_))
    print(object_)
    # y 
    object_ = bcutils.load_object(object_path='/home/bruno/Dropbox/utfpr/tcc2/one_hot_labels_list.pkl')
    print(type(object_))
    print(object_)
    # graph
    object_ = bcutils.load_object(object_path='/home/bruno/Dropbox/utfpr/tcc2/graph.pkl')
    print(type(object_))
    print(object_)
    
    

    


