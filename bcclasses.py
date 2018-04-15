class DetailedBoundBox:
    """
        DelailedBoundBox is a helper class to facilitate the manipulation of data. 
        It's attributes are: 
            global_class => one of those values: ['Person', 'Animal', 'Vehicle', 'Indoor']
            global_class => one of those values: ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tv/monitor']
            bndbox_coordinates => the coordinates of bound box. Something like that (98, 58, 202, 225)
            image_id => Image's id of which boundbox belongs.
    """
    def __init__(self, global_class, sub_class, bndbox_coordinates, image_id):
        self.global_class = global_class
        self.sub_class = sub_class
        self.bndbox_coordinates = bndbox_coordinates
        self.image_id = image_id

    def __str__(self):
        str_ = 'Global Class: {}\nSub Class: {}\nBndBox Coordinates: {}\nImage ID: {}'.format(self.global_class, self.sub_class, self.bndbox_coordinates, self.image_id)
        return str_

class Sources:
    def __init__(self, database, annotation, image):
        self.database = database
        self.annotation = annotation
        self.image = image


class Sizes():
    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth


class Bndbox:
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


class Obj:
    def __init__(self, name, pose, difficult, truncated, bndbox):
        self.name = name
        self.pose = pose
        self.difficult = difficult
        self.truncated = truncated
        self.bndbox = bndbox

    def get_bndbox(self):
        return int(self.bndbox.xmin), int(self.bndbox.ymin), int(self.bndbox.xmax), int(self.bndbox.ymax)

    def __str__(self):
        return """name: {}\npose:{}\ndifficult:{}\ntruncated: {}\nbndbox:\n\txmin:{}\n\tymin:{}\n\txmax:{}\n\tymax:{}\n\t""".format(
            self.name, self.pose, self.difficult, self.truncated, self.bndbox.xmin, self.bndbox.ymin, self.bndbox.xmax,
            self.bndbox.ymax)


class Annotation:
    def __init__(self, folder, filename, sources, sizes, segmented, objects):
        self.folder = folder
        self.filename = filename
        self.sources = sources
        self.sizes = sizes
        self.segmented = segmented
        self.objects = objects