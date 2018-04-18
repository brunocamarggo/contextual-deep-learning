#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
from bcloader import *
import scipy
import numpy
import collections
import config

DATA_DIR = config.paths['DATA_DIR']


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
                bottleneck_name = bottlenecks[index].split('_')[3].replace('.jpg', '').replace('.txt', '')
                jpg_img_name = bndboxes_jpg[index].split('_')[0] + '_' + bndboxes_jpg[index].split('_')[1]
                jpg_bb_class = bndboxes_jpg[index].split('_')[3].replace('.jpg', '')
                test = (jpg_bb_class == obj.name == bottleneck_name == graph_labels[index])
                
                if not test:
                    result = False
                    print('{} = {} and {} = {} and {} = {}[{}]'.format(jpg_img_name, xml.filename, obj.name,
                                                                       jpg_bb_class, bottleneck_name, obj.name, test))
                index += 1   
        
        self.assertTrue(result)

    def test_input_types(self):
        input_files = [f for f in listdir(DATA_DIR) if isfile(join(DATA_DIR, f))]
        result = True
        for f in input_files:
            if f.endswith('.x'):
                object_ = load_object(object_path=join(DATA_DIR, f))
                if type(object_) != scipy.sparse.csr.csr_matrix:
                    result = False
            elif f.endswith('.tx'):
                object_ = load_object(object_path=join(DATA_DIR, f))
                if type(object_) != scipy.sparse.csr.csr_matrix:
                    result = False
            elif f.endswith('.allx'):
                object_ = load_object(object_path=join(DATA_DIR, f))
                if type(object_) != scipy.sparse.csr.csr_matrix:
                    result = False
            elif f.endswith('.y'):
                object_ = load_object(object_path=join(DATA_DIR, f))
                if type(object_) != numpy.ndarray:
                    result = False
            elif f.endswith('.ty'):
                object_ = load_object(object_path=join(DATA_DIR, f))
                if type(object_) != numpy.ndarray:
                    result = False
            elif f.endswith('.ally'):
                object_ = load_object(object_path=join(DATA_DIR, f))
                if type(object_) != numpy.ndarray:
                    result = False
            elif f.endswith('.graph'):
                object_ = load_object(object_path=join(DATA_DIR, f))
                if type(object_) != collections.defaultdict:
                    result = False
            elif f.endswith('.test.index'):
                object_ = load_object(object_path=join(DATA_DIR, f))
                if type(object_) != list:
                    result = False
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
