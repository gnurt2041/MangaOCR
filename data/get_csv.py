import manga109api
import pandas as pd
import cv2 
import os
from tqdm import tqdm

import sys
sys.path.append("..")

from MangaOCR.env import MANGA109_ROOT

def create_csv(path, output_csv_name):
    # Read annotations by manga109api
    # manga109_root_dir = "E:\Manga109 dataset\Manga109s_released_2021_12_30"
    manga109_root_dir = path
    manga109 = manga109api.Parser(root_dir=manga109_root_dir)

    # Create img and text files 
    csv_file = []
    text_img = {}
    for book in tqdm(manga109.books):
        a = manga109.get_annotation(book = book)["page"]
        for index, i in enumerate(a):
            text_annotation = i["text"]
            if text_annotation != []:
                img_path = manga109.img_path(book= book, index=index)
                for text in text_annotation:
                    text_img["text"] = text["#text"]
                    text_img["xmin"] = text["@xmin"]
                    text_img["ymin"] = text["@ymin"]
                    text_img["xmax"] = text["@xmax"]
                    text_img["ymax"] = text["@ymax"]
                    text_img["img_path"] = img_path
                    # print(img_path)
                    # print(text)
                    # print(text["#text"])
                    # print(text["@xmin"], text["@ymin"], text["@xmax"], text["@ymax"]) 
                    csv_file.append(text_img)
                    text_img = {}

    data_file = pd.DataFrame(csv_file)
    # Specify the file path for the CSV file
    # csv_file_path = "data_text_img.csv"
    csv_file_path = output_csv_name
    # Save the DataFrame to a CSV file
    data_file.to_csv(csv_file_path, index=False)
    print("CSV file created successfully!")


def create_img_text(csv_path, margin):
    # Read csv file 
    # data = pd.read_csv("data_text_img.csv")
    data = pd.read_csv(csv_path)
    img_path = data.img_path
    text = data.text
    xmin = data.xmin
    xmax = data.xmax
    ymin = data.ymin
    ymax = data.ymax

    csv_new_file = []
    img_text_dict = {}
    name = ""
    if not os.path.exists(f"{MANGA109_ROOT}/images_text/"):
        os.mkdir(f"{MANGA109_ROOT}/images_text/")
    for i, path in enumerate(tqdm(img_path)):
        # print(path)
        image = cv2.imread(path)
        xmin_img = max(xmin[i] - margin, 0)
        xmax_img = min(xmax[i] + margin, image.shape[1])
        ymin_img = max(ymin[i] - margin, 0)
        ymax_img = min(ymax[i] + margin, image.shape[0])
        img_text = image[ymin_img : ymax_img, xmin_img : xmax_img]
        name = f"{MANGA109_ROOT}/images_text/" + str(i) + ".jpg"
        # print(name, text[i])
        cv2.imwrite(name, img_text)
        img_text_dict["text"] = text[i]
        img_text_dict["path"] = name
        csv_new_file.append(img_text_dict)
        img_text_dict = {}

    data = pd.DataFrame(csv_new_file)
    data.to_csv('text_img.csv', index = False)


if __name__ == '__main__':
    # path = "E:\Manga109 dataset\Manga109s_released_2021_12_30"
    output_csv_name = "data_text_img.csv"
    margin = 10
    create_csv(MANGA109_ROOT, output_csv_name)
    create_img_text(output_csv_name, margin)




