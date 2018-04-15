from bcloader import *
import random


def generate_random_dataset(size=None):
    xmls = load_xmls()
    print(len(xmls))
    size_ = 0
    dataset = []
    already_chosen = []
    while size_ != size:
        random_index = random.choice(range(len(xmls)))
       
        if random_index not in already_chosen:
            already_chosen.append(random_index)
            data = {
                'random_index': random_index,
                'xml': xmls[random_index]
            }
            dataset.append(data)
            size_ += 1

    sub_classes = []
    for data in dataset:
        for obj in data['xml'].objects:
            sub_classes.append(obj.name)
    counter = dict()
    k = 0
    for sub_class in sub_classes:
        k += 1
        if not sub_class in counter:
            counter[sub_class] = 1
        else:
            counter[sub_class] += 1
    print(counter)
    print(len(counter))
    print(k)


if __name__ == '__main__':
    generate_random_dataset(size=2000)

