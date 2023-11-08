import argparse
import json
import os
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[1, 10], help="Range of seeds"
    )
    args = parser.parse_args()
    return args


def generate_seeds(args):
    data_path = "/home/aeen/fewshot/Datasets/dota/dotasplit/datasplit/trainvalno5k.json"
    data = json.load(open(data_path))

    new_all_cats = []
    for cat in data["categories"]:
        new_all_cats.append(cat)

    id2img = {}
    for i in data["images"]:
        id2img[i["id"]] = i

    anno = {i: [] for i in ID2CLASS.keys()}
    for a in data["annotations"]:
        if a["iscrowd"] == 1:
            continue
        anno[a["category_id"]].append(a)

    for i in range(args.seeds[0], args.seeds[1]):
        random.seed(i)
        for c in ID2CLASS.keys():
            img_ids = {}
            for a in anno[c]:
                if a["image_id"] in img_ids:
                    img_ids[a["image_id"]].append(a)
                else:
                    img_ids[a["image_id"]] = [a]

            sample_shots = []
            sample_imgs = []
            for shots in [1, 2, 3, 5, 10, 30]:
                while True:
                    imgs = random.sample(list(img_ids.keys()), shots)
                    for img in imgs:
                        skip = False
                        for s in sample_shots:
                            if img == s["image_id"]:
                                skip = True
                                break
                        if skip:
                            continue
                        if len(img_ids[img]) + len(sample_shots) > shots:
                            continue
                        sample_shots.extend(img_ids[img])
                        sample_imgs.append(id2img[img])
                        if len(sample_shots) == shots:
                            break
                    if len(sample_shots) == shots:
                        break
                new_data = {
#                     "info": data["info"],
#                     "licenses": data["licenses"],
                    "images": sample_imgs,
                    "annotations": sample_shots,
                }
                save_path = get_save_path_seeds(
                    data_path, ID2CLASS[c], shots, i
                )
                new_data["categories"] = new_all_cats
                
                with open(save_path, "w") as f:
                    json.dump(new_data, f)


def get_save_path_seeds(path, cls, shots, seed):
    prefix = "full_box_{}shot_{}_trainval".format(shots, cls)
    save_dir = os.path.join("/home/aeen/fewshot/Datasets/dota", "dotasplit", "seed" + str(seed))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, prefix + ".json")
    return save_path

if __name__ == "__main__":
    ID2CLASS = {
        1: "plane",
        2: "baseball-diamond",
        3: "bridge",
        4: "ground-track-field",
        5: "small-vehicle",
        6: "large-vehicle",
        7: "ship",
        8: "tennis-court",
        9: "basketball-court",
        10: "storage-tank",
        11: "soccer-ball-field",
        12: "roundabout",
        13: "harbor",
        14: "swimming-pool",
        15: "helicopter",
        16: "container-crane",
    }
    CLASS2ID = {v: k for k, v in ID2CLASS.items()}

    args = parse_args()
    generate_seeds(args)
