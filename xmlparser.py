import xml.etree.ElementTree as ET
from skimage import io
from skimage import transform
import cv2
from bcclasses import (
    Sources,
    Sizes,
    Bndbox,
    Obj,
    Annotation
)

def imshow(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def build(xml):
    root = ET.parse(xml).getroot()
    folder = root.find('folder').text
    filename = root.find('filename').text

    for source in root.findall('source'):
        database = source.find('database').text
        annotation = source.find('annotation').text
        image = source.find('image').text
        sources = Sources(database, annotation, image)

    sizes = []
    for size in root.findall('size'):
        width = size.find('width').text
        height = size.find('height').text
        depth = size.find('depth').text
        sizes = Sizes(width, height, depth)

    segmented = root.find('segmented').text

    objects = []

    for obj in root.findall('object'):
        name = obj.find('name').text
        pose = obj.find('pose').text
        truncated = ''
        if obj.find('truncated'):
            truncated = obj.find('truncated').text
        difficult = obj.find('difficult').text
        xmin = float(obj.find('bndbox').find('xmin').text)
        ymin = float(obj.find('bndbox').find('ymin').text)
        xmax = float(obj.find('bndbox').find('xmax').text)
        ymax = float(obj.find('bndbox').find('ymax').text)
        bndbox = Bndbox(xmin, ymin, xmax, ymax)
        obj = Obj(name, pose, truncated, difficult, bndbox)
        objects.append(obj)

    annotation = Annotation(folder, filename, sources, sizes, segmented, objects)

    return annotation


if __name__ == '__main__':
    # 2007_000042
    # 2008_003608
    annotation = build('/home/bruno/Documents/VOCdevkit/VOC2012/Annotations/2007_000042.xml')
    image = io.imread('/home/bruno/Documents/VOCdevkit/VOC2012/JPEGImages/2007_000042.jpg')
    imshow("", image)
    # io.imread(image)
    print("Number of objects: {}".format(len(annotation.objects)))
    for obj in annotation.objects:
        bnd = obj.get_bndbox()
        x1, y1, x2, y2 = obj.get_bndbox()
        cropped = image[y1:y2, x1:x2]
        cropped = transform.resize(cropped, (224, 224))
        imshow(annotation.objects[0].name, cropped)
