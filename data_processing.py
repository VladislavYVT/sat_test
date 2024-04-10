from math import ceil

import numpy as np
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision.transforms import v2
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import os

from SiameseDataset import SiameseDataset


def create_correct_set(arr, uav, sat):
    return np.array([[uav + x, sat + x, 1] for x in arr])


def create_correct_false_pairs(cvals, fvals, scale_factor, pref1, pref2):
    from_correct = np.array(list(cvals) * scale_factor)
    fvalues = np.array(list(fvals) * (int(len(from_correct) / len(fvals)) + 1))
    # To make selection more smooth, same item can't be selected more than 1 time on the current dataset,
    # so it forces to select different ones
    random_choices = np.random.choice(fvalues, len(from_correct), replace=False)
    return np.array([[pref1 + from_correct[i], pref2 + random_choices[i], 0] for i in range(len(random_choices))])


def get_df_split(carr, farr, scale_one_correct, uav, sat, fsat):
    print("creating correct")
    correct = create_correct_set(carr, uav, sat)
    print("correct_number " + str(len(correct)))
    print("creating incorrect2")
    incorrect2 = create_correct_false_pairs(carr, farr, scale_one_correct, uav, fsat)
    print("icorrect2 number " + str(len(incorrect2)))
    incorrect3 = create_correct_false_pairs(carr, farr, scale_one_correct, sat, fsat)
    print("icorrect3 number " + str(len(incorrect3)))
    print("Concatenating")
    resulting_array = np.concatenate((correct, incorrect2, incorrect3))
    print("concatenated_number " + str(len(resulting_array)))
    return pd.DataFrame(resulting_array, columns=["path1", "path2", "label"])


def get_df_split_upscaled(carr, farr, scale_one_correct, uav, sat, fsat):
    print("creating correct")
    correct = create_correct_set(carr, uav, sat)
    print("correct_number " + str(len(correct)))
    print("creating incorrect1")
    incorrect2 = create_correct_false_pairs(carr, farr, scale_one_correct, uav, fsat)
    print("icorrect2_number " + str(len(incorrect2)))
    incorrect3 = create_correct_false_pairs(carr, farr, scale_one_correct, sat, fsat)
    print("icorrect3_number " + str(len(incorrect3)))
    print("Concatenating")
    resulting_array = np.concatenate((correct, correct, incorrect2, incorrect3))
    print("concatenated_number " + str(len(resulting_array)))
    return pd.DataFrame(resulting_array, columns=["path1", "path2", "label"])


def get_common_transform():
    return v2.Compose([
        v2.ToPILImage(),
        v2.Resize((224, 224)),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_augment_transform():
    return v2.Compose([
        v2.ToPILImage(),
        v2.Resize((224, 224)),
        v2.RandomResizedCrop(size=(224, 224), scale=(0.6, 1.0)),
        v2.ToTensor(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(degrees=(0, 180)),
        v2.RandomPerspective(distortion_scale=0.2, p=0.5),
        v2.ColorJitter(brightness=0.2,
                       contrast=0.1,
                       saturation=0.2,
                       hue=0.1),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def preprocess_and_save_dataset(path, dataloader, repeat_factor=1):
    counter = 0
    for _ in range(repeat_factor):
        for i, data in enumerate(dataloader):
            with open(path + str(counter) + ".pkl", 'wb') as outp:
                pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)
                counter += 1
            inverted_data = [data[1], data[0], data[2]]
            with open(path + str(counter) + ".pkl", 'wb') as outp:
                pickle.dump(inverted_data, outp, pickle.HIGHEST_PROTOCOL)
                counter += 1
            print("Finished writing file number %d" % (counter - 1))


def prep_img(img_path):
    img = read_image(img_path)
    transform = v2.Compose([
        v2.ToPILImage(),
        v2.Resize((224, 224)),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_res = transform(img)
    return img, img_res.unsqueeze(0)


def preprocess_datasets(uav="dataset/RGB/uav_images/", sat="dataset/RGB/sat_images/",
                        fsat="dataset/RGB/false_sat_images/", repeat_factor = 4):
    seed = 13
    np.random.seed(seed)
    correct_names_sat = np.array(os.listdir(sat))
    false_names = np.array(os.listdir(fsat))
    ctrain, ctest = train_test_split(correct_names_sat, test_size=0.2, shuffle=True, random_state=seed)
    ftrain, ftest = train_test_split(false_names, test_size=0.2, shuffle=True, random_state=seed)
    df_train_upscaled = get_df_split_upscaled(ctrain, ftrain, 3, uav, sat, fsat)

    batch_size = 32
    dataset_augmented = SiameseDataset(df_train_upscaled.values, transform=get_augment_transform())
    dataloader_augmented = DataLoader(dataset_augmented, batch_size=batch_size, shuffle=True)
    expected_output_files = ceil(len(df_train_upscaled.index) / 32) * 2 * repeat_factor
    print("Number of expected files in the output %d" % expected_output_files)
    preprocess_and_save_dataset("output/processed_dataset_augmented/", dataloader_augmented, repeat_factor)
    df_test = get_df_split(ctest, ftest, 3, uav, sat, fsat)
    df_test.to_csv("df_test.csv")
    eval_set = SiameseDataset(df_test.values, transform=get_common_transform())
    dataloader_eval = DataLoader(eval_set, batch_size=batch_size, shuffle=False)
    preprocess_and_save_dataset("output/processed_eval/", dataloader_eval, 1)


if __name__ == "__main__":
    preprocess_datasets()
