"""
Author:
-------
- Amos Matter (mail@amosmatter.ch)

License:
--------
- MIT License

"""


from pathlib import Path
import random
import shutil
import os

import numpy as np


def split_dataset(
    valid_classes,
    dataset_path,
    train_path,
    test_path,
    val_path,
    train_ratio=0.8,
    test_ratio=0.18,
    val_ratio=0.02,
    n_imgs_per_place=4,
    n_places=1400,
):

    dataset_path = Path(dataset_path)
    train_path = Path(train_path)
    test_path = Path(test_path)
    val_path = Path(val_path)

    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    for folder in valid_classes:
        files = os.listdir(dataset_path / folder)
        file_data = {}

        for file in files:
            place_id = int(file.split("_")[1])
            file_data.setdefault(place_id, []).append(file)

        n_lens = np.zeros(max(file_data.keys()) + 1)
        for place_id, files in file_data.items():
            n_lens[place_id] = len(files)

        amount, n_with_amount = np.unique(n_lens, return_counts=True)

        n_usable_places = np.sum(n_with_amount[amount >= n_imgs_per_place])

        if n_usable_places < n_places:
            continue

        place_ids = np.argwhere(n_lens >= n_imgs_per_place).reshape(-1)

        np.random.shuffle(place_ids)
        split_i1 = int(n_places * train_ratio)
        split_i2 = int(n_places * (train_ratio + test_ratio))
        train_place_ids = place_ids[:split_i1]
        test_place_ids = place_ids[split_i1:split_i2]
        val_place_ids = place_ids[split_i2:n_places]

        (train_path / folder).mkdir(parents=True, exist_ok=True)
        (test_path / folder).mkdir(parents=True, exist_ok=True)
        (val_path / folder).mkdir(parents=True, exist_ok=True)

        for place_id in train_place_ids:

            files = random.sample(file_data[place_id], n_imgs_per_place)
            for file in files:
                shutil.copy(dataset_path / folder / file, train_path / folder / file)

        for place_id in test_place_ids:
            files = random.sample(file_data[place_id], n_imgs_per_place)
            for file in files:
                shutil.copy(dataset_path / folder / file, test_path / folder / file)

        for place_id in val_place_ids:
            files = random.sample(file_data[place_id], 1)
            for file in files:
                shutil.copy(dataset_path / folder / file, val_path / folder / file)


if __name__ == "__main__":
    PROJ_PATH = Path(__file__).resolve().parent
    DATASET_PATH = PROJ_PATH / "archive" / "Images"
    TRAIN_PATH = PROJ_PATH / "train"
    TEST_PATH = PROJ_PATH / "test"
    VAL_PATH = PROJ_PATH / "val"

    print(
        split_dataset(
            os.listdir(DATASET_PATH), DATASET_PATH, TRAIN_PATH, TEST_PATH, VAL_PATH
        )
    )
