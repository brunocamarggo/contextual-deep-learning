from os import path
import xmlparser
from os import listdir
from os.path import isfile, join
import skimage
import config

IMG_DIR = config.paths['IMG_DIR_MIT67']

def prepare_image(img):
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img

if '__main__' == __name__:
    images = [img for img in listdir(IMG_DIR)]
    for img in images:
        img_name = img
        img = skimage.io.imread(join(IMG_DIR, img))
        img = prepare_image(img)
        skimage.io.imsave(path.join(IMG_DIR, img_name), img)
    print('DONE')
