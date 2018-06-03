import xml.etree.ElementTree as ET
from os import listdir
from bcutils import get_global_class_MIT67


class MIT67Annotation:
    def __init__(self, folder='', filename='', source=None, objects=None, scenedescription=None):
        self.folder = folder
        self.filename = filename
        self.source = source
        self.objects = objects
        self.scenedescription = scenedescription

    def __str__(self):
        str_ = '--- MIT67 Annotation ---\nFolder: {}\nFilename: {}\nSource: {}, Objects: {}, Scenedescription: {}'.\
            format(self.folder, self.filename, self.source, self.objects, self.scenedescription)
        return str_

    def get_bound_box(self):
        return None


def buildMIT67(xml):
    ann = None
    try:
        root = ET.parse(xml).getroot()

        folder = root.find('folder').text.strip()
        filename = root.find('filename').text.strip()

        source_image = root.find('source').find('sourceImage').text.strip()
        source_annotation = root.find('source').find('sourceAnnotation').text.strip()
        source = {
            'sourceImage': source_image,
            'sourceAnnotation': source_annotation
        }
        object_list = []
        id = None
        scenedescription = None
        for object_ in root.findall('object'):
            name = object_.find('name').text.strip()
            deleted = object_.find('deleted').text.strip()
            date = object_.find('date').text.strip()

            try:
                id = object_.find('id').text.strip()
                scenedescription = root.find('scenedescription').text.strip()
            except:
                id = None
                scenedescription = None

            username = object_.find('polygon').find('username').text.strip()
            pts = []
            for pt in object_.find('polygon').findall('pt'):
                x = int(pt.find('x').text.strip())
                y = int(pt.find('y').text.strip())
                pts.append((x, y))


            polygon = {
                'username': username,
                'pts': pts
            }

            obj = {
                'name': name,
                'deleted': deleted,
                'date': date,
                'id': id,
                'polydon': polygon
            }
            object_list.append(obj)


        ann = MIT67Annotation(folder=folder, filename=filename, source=source, objects=object_list,
                              scenedescription=scenedescription)
    except:
        print('------> Err: {}'.format(xml))
    return ann





if __name__ == '__main__':
    # ------> Err: / home / bruno / Documents / MIT67 / Annotations / kitchen / dsc01464.xml
    # ------> Err: / home / bruno / Documents / MIT67 / Annotations / kitchen / dscn8004.xml
    # ------> Err: / home / bruno / Documents / MIT67 / Annotations / bathroom / indoor_0124.xml
    # ------> Err: / home / bruno / Documents / MIT67 / Annotations / mall / Buenos_Aires_shopping_center_2_.xml

    path = '/home/bruno/Documents/MIT67/Annotations/'
    dirs = [dir_ for dir_ in listdir(path)]
    num_obj = 0
    num_images = 0
    classes = {}
    for i, val in enumerate(dirs):
        for xml in listdir(path+dirs[i]):
            f = path + dirs[i] + '/' + xml
            ann = buildMIT67(f)
            if ann:
                num_images += 1
                for obj in ann.objects:
                    # print(obj['name'])
                    num_obj += 1
                global_class = get_global_class_MIT67(subclass=ann.folder)
                if global_class not in classes:
                    classes[global_class] = 1
                else:
                    classes[global_class] += 1

    print(num_images)
    print(num_obj)
    print(classes)
