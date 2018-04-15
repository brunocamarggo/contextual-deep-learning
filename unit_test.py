import unittest
from bcloader import *


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


if __name__ == '__main__':
    unittest.main()
