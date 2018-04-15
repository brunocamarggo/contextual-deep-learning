from itertools import permutations
from os import path
import xmlparser
from os import listdir
from os.path import isfile, join
import skimage


XML_DIR = "/home/bruno/Documents/VOCdevkit/VOC2012/Annotations/"
IMG_DIR = "/home/bruno/Documents/VOCdevkit/VOC2012/JPEGImages/"
BOUND_BOX_PATH = "/home/bruno/Documents/VOCdevkit/VOC2012/BoundBoxes"


def prepare_image(img):
    # load image
    #img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img


def imcrop(image, obj):
    x1, y1, x2, y2 = obj.get_bndbox()
    cropped = image[y1:y2, x1:x2]
    return prepare_image(cropped)


if __name__ == '__main__':
    all_xml_files = [xml_file for xml_file in listdir(XML_DIR) if isfile(join(XML_DIR, xml_file))]

    for index, xml_file in enumerate(all_xml_files):
        # print("checking {}...".format(xml_file))
        ann = xmlparser.build(XML_DIR + xml_file)
        if len(ann.objects) >= 2:
            img_name = xml_file.replace('.xml', '.jpg')
            for i, obj in enumerate(ann.objects):
                image = skimage.io.imread(IMG_DIR + img_name)
                cropped = imcrop(image, obj)
                cropped_name = xml_file.replace('.xml', '') + '_' + "{:0>2}".format(i) +'_' + obj.name + '.jpg'
                skimage.io.imsave(path.join(BOUND_BOX_PATH, cropped_name), cropped)
                # print('cropped ' + cropped_name)

       

