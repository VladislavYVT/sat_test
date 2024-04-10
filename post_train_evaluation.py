import os

import numpy as np
import torch

from SiameseNet import SiameseResNet
from data_processing import get_common_transform
from torchvision.io import read_image


# This is done purely for exploration of results and demonstration purposes
def evaluate_same_names(model, uav_path="dataset/RGB/uav_images/",
                        sat_path="dataset/RGB/sat_images/", index_range=(267, 691)):
    #doesn't use dataloader, as
    counter = 0
    for k in range(index_range[0], index_range[1]):
        fname = "DJI_0" + str(k) + ".JPG"
        s = get_image_processed(sat_path + fname)
        u = get_image_processed(uav_path + fname)
        sim = get_model_prediction(model, u[1], s[1])
        if sim < 0.5:
            print(k)
            print(sim)
            counter += 1
    return counter


def get_model_prediction(model, tensor1, tensor2):
    with torch.no_grad():
        out = model(tensor1, tensor2)
        return out[0][0]


def get_image_processed(img_path):
    transform = get_common_transform()
    img = read_image(img_path)
    img_res = transform(img)
    return img, img_res.unsqueeze(0)

def evaluate_against_fsat_randoms(model, path_1, fsat_path="dataset/RGB/false_sat_images/"):
    correct_names = np.array(os.listdir(path_1))
    fsat_names = np.array(os.listdir(fsat_path))
    counter = 0;
    print("CHECKING FOR INCORRECT CLASSIFICATIONS")
    random_choices = np.random.choice(fsat_names, len(correct_names), replace=False)
    for k in range(len(correct_names)):
        u = get_image_processed(path_1 + correct_names[k])
        s = get_image_processed(fsat_path + random_choices[k])
        sim = get_model_prediction(model, u[1], s[1])
        if sim > 0.5:
            print(correct_names[k])
            print(random_choices[k])
            print(sim)
            counter += 1

    return counter


def load_model(path="output/siamese_model.pt"):
    model_dict = torch.load(path)
    model = SiameseResNet()
    model.load_state_dict(model_dict)
    model.eval()
    return model


if __name__ == "__main__":
    uav = "dataset/RGB/uav_images/"
    sat = "dataset/RGB/sat_images/"
    fsat = "dataset/RGB/false_sat_images/"
    model = load_model()
    counter_1 = evaluate_same_names(model, uav, sat)
    counter_2 = evaluate_against_fsat_randoms(model, uav, fsat)
    print("number of misclassifications on correct %d" % counter_1)
    print("number of misclassifications on incorrect %d" % counter_2)