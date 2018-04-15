from os import listdir
from os.path import isfile, join
import xmlparser
from igraph import *
import config
import pickle

BOUND_BOX_PATH = config.paths['BOUND_BOX_PATH']
XML_DIR = config.paths['XML_DIR']
BOTTLENECK_PATH = config.paths['BOTTLENECK_PATH']
GRAPH_FILE = config.paths['GRAPH_FILE']


def load_xmls(path=XML_DIR):
    """
    Returns all xml (as an Annotation object) present in a directory.
    """
    print('Loading xmls files. Please wait...')
    all_xmls = [xmlparser.build(path + xml) for xml in sorted(listdir(path)) if isfile(join(path, xml))]
    xmls = [xml for xml in all_xmls if len(xml.objects) >= 2]
    print('\t[DONE]')
    return xmls


def load_bottlenecks(path=BOTTLENECK_PATH):
    """
    Returns a list of all bound boxes bottlenecks already produced. 
    Look to produce_bottlenecks.py for more information.
    """
    print('Loading bottlenecks files. Please wait...')
    bottlenecks = [bottleneck for bottleneck in sorted(listdir(path)) if isfile(join(path, bottleneck))]
    print('\t[DONE]')
    return bottlenecks


def load_bndboxes_jpg(path=BOUND_BOX_PATH):
    """
    Returns a list of all bound boxes as jpg file already produced. 
    Look to produce_all_bound_boxes_image_files.py for more information.
    """
    print('Loading bound boxes files. Please wait...')
    bndboxes_jpg = [bb for bb in sorted(listdir(path)) if isfile(join(path, bb))]
    print('\t[DONE]')
    return bndboxes_jpg


def load_graph_labels(graph_file=GRAPH_FILE):
    """
    Returns a list of all vertex labels of a graph
    """
    print('Loading graph labels. Please wait...')
    graph = Graph.Read_GraphMLz(graph_file)
    print('\t[DONE]')
    return graph.vs['label']

def load_object(object_path=None):
    """
    Loads an objects in pickle format.
    """
    with open(object_path, 'rb') as input_:
        graph_dict = pickle.load(input_)
        return graph_dict

        

