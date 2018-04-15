import unittest
from bcloader import *
import scipy
import numpy
import collections


class TestSequentialInput(unittest.TestCase):
    def test_sequential_input_bound_boxes(self):
        """
        This test checks with the bound boxes are in the same sequential (by comparing the class):
            - bndboxes_jpg: list of all bound boxes as jgp file
            - bottlenecks: list of all bound boxes bottlenecks files
            - xmls: list of all xml annotations with 2 of more bound boxes
            - graph_labels: list of all vertex labels of the graph
        
        bndboxes_jpg[index] == bottlenecks[index] == xmls.xml.objects[index] == graph_labels
        == must be the same sub class.
        """
        result = True
        # loading all lists
        bndboxes_jpg = load_bndboxes_jpg()
        bottlenecks = load_bottlenecks()       
        graph_labels = load_graph_labels()
        xmls = load_xmls()
        index = 0
        for xml in xmls:
            for obj in xml.objects:
                if len(graph_labels) == index:
                    break
                bottleneck_name = bottlenecks[index].split('_')[3].replace('.jpg','').replace('.txt', '')
                jpg_img_name = bndboxes_jpg[index].split('_')[0] + '_'+ bndboxes_jpg[index].split('_')[1]
                jpg_bb_class = bndboxes_jpg[index].split('_')[3].replace('.jpg','')
                test = (jpg_bb_class == obj.name == bottleneck_name == graph_labels[index])
                
                if not test:
                    result = False
                    print('{} = {} and {} = {} and {} = {}[{}]'.format(jpg_img_name, xml.filename, obj.name,
                                                                       jpg_bb_class, bottleneck_name, obj.name, test))
                index += 1   
        
        self.assertTrue(result)

    def test_input_types(self):
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
        result = True
        object_ = load_object(object_path='/home/bruno/Dropbox/utfpr/tcc2/data/ind.voc2012.x')
        if type(object_) != scipy.sparse.csr.csr_matrix:
            result = False
        # tx
        object_ = load_object(object_path='/home/bruno/Dropbox/utfpr/tcc2/data/ind.voc2012.tx')
        if type(object_) != scipy.sparse.csr.csr_matrix:
            result = False
        # y
        object_ = load_object(object_path='/home/bruno/Dropbox/utfpr/tcc2/data/ind.voc2012.y')
        if type(object_) != numpy.ndarray:
            result = False
        # ty
        object_ = load_object(object_path='/home/bruno/Dropbox/utfpr/tcc2/data/ind.voc2012.ty')
        if type(object_) != numpy.ndarray:
            result = False
        # graph
        object_ = load_object(object_path='/home/bruno/Dropbox/utfpr/tcc2/data/ind.voc2012.graph')
        if type(object_) != collections.defaultdict:
            result = False
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()
