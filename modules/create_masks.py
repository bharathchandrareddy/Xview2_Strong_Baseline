# using same logic as legacy/create_masks.py

import json
from shapely.wkt import loads
from shapely.geometry import shape
from multiprocessing import Pool
import sys
from os import path, makedirs, listdir
import timeit
import cv2
import random
import numpy as np
import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

np.random.seed(1)
random.seed(1)
sys.setrecursionlimit(10000)

masks_dir = 'masks'

# Update the train directories with your paths
train_dirs = ["C:\\Users\\PC\\Desktop\\damage_assessement_data\\train",
              "C:\\Users\\PC\\Desktop\\damage_assessement_data\\tier3",
              "C:\\Users\\PC\\Desktop\\damage_assessement_data\\test"]

def mask_for_polygon(poly, im_size=(1024, 1024)):
    img_mask = np.zeros(im_size, np.uint8)
    def int_coords(x): return np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords)]
    interiors = [int_coords(pi.coords) for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask

damage_dict = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": 1  # ?
}

def process_image(json_file):
    js1 = json.load(open(json_file))
    js2 = json.load(open(json_file.replace('_pre_disaster', '_post_disaster')))

    msk = np.zeros((1024, 1024), dtype='uint8')
    msk_damage = np.zeros((1024, 1024), dtype='uint8')



    for feat in js1['features']['xy']:
        poly = loads(feat['wkt'])
        _msk = mask_for_polygon(poly)
        msk[_msk > 0] = 255

    for feat in js2['features']['xy']:
        poly = loads(feat['wkt'])
        subtype = feat['properties']['subtype']
        _msk = mask_for_polygon(poly)
        msk_damage[_msk > 0] = damage_dict[subtype]
        

    pre_mask_path = json_file.replace('\\labels\\', '\\masks\\').replace('_pre_disaster.json', '_pre_disaster.png')
    post_mask_path = json_file.replace('\\labels\\','\\masks\\').replace('_pre_disaster.json', '_post_disaster.png')
    cv2.imwrite(pre_mask_path, msk, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite(post_mask_path, msk_damage, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    

if __name__ == '__main__':
    t0 = timeit.default_timer()

    all_files = []
    for d in train_dirs:
        makedirs(path.join(d, masks_dir), exist_ok=True)
        for f in sorted(listdir(path.join(d, 'images'))):
            if '_pre_disaster.png' in f:
                label_path = path.join(d,'labels',f.replace('_pre_disaster.png','_pre_disaster.json'))
                all_files.append(label_path)

    total_damage_count = {
        "no-damage": 0,
        "minor-damage": 0,
        "major-damage": 0,
        "destroyed": 0,
        "un-classified": 0
    }

    with Pool() as pool:
        damage_counts = pool.map(process_image, all_files)

    # Aggregate the counts from all files
    for count in damage_counts:
        for key in total_damage_count:
            total_damage_count[key] += count[key]

    print('Total damage counts across all files:', total_damage_count)
    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
