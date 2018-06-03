from os import listdir
from bcutils import get_global_class_MIT67
import config

DATASET_DIR = config.paths['IMG_DIR_MIT67']
BOTTLENECKS_DIR = config.paths['BOTTLENECK_PATH_MIT67']

def evaluate():
    images = [img for img in sorted(listdir(DATASET_DIR))]
    bottlenecks = [b for b in sorted(listdir(BOTTLENECKS_DIR))]
    for i, img in enumerate(images):
        if not bottlenecks[i].replace('.txt', '') == img:
            raise Exception('Err')

    dataset_dict = {}
    for i, img in enumerate(images):
        class_ = img.split('*')[0]
        class_ = get_global_class_MIT67(subclass=class_)
        images[i] = class_ + '*' + img
        bottlenecks[i] = class_ + '*' + bottlenecks[i]
        if class_ not in dataset_dict:
            dataset_dict[class_] = 1
        else:
            dataset_dict[class_] += 1

    print(dataset_dict)
    return sorted(images), sorted(bottlenecks), dataset_dict


if __name__ == '__main__':
    evaluate()