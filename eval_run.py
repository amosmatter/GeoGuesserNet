"""
Author:
-------
- Amos Matter (mail@amosmatter.ch)

License:
--------
- MIT License

"""

# %%
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
import torch
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.models.efficientnet import efficientnet_v2_s, EfficientNet_V2_S_Weights
import msvcrt

import keyboard
import time

from model import torch_persistent_model
from eval_functions import format_result, test_model
import pandas as pd


def parse_logs(log_path: Path):
    bigdf = pd.DataFrame()
    for f in log_path.glob("*.csv"):
        print(f)
        df = pd.read_csv(f)
        bigdf = pd.concat([bigdf, df])

    bigdf.sort_values(by="timestamp", inplace=True)

    biglbl = ";".join(bigdf["labels"])
    bigpred = ";".join(bigdf["preds"])

    biglbl = np.array(
        [
            np.fromstring(biglbl, dtype=int, sep=";"),
            np.fromstring(bigpred, dtype=int, sep=";"),
        ]
    )
    print(biglbl)
    window_sz = len(bigdf) // 100
    row_sz = 16 * window_sz
    n_rows = biglbl.shape[1] // row_sz
    biglbl = biglbl[:, : n_rows * row_sz]

    f1 = []
    for i in range(n_rows):
        lbl = biglbl[0, i * row_sz : (i + 1) * row_sz]
        pred = biglbl[1, i * row_sz : (i + 1) * row_sz]
        f1.append(f1_score(lbl, pred, average="macro", zero_division=0))

    return bigdf, f1


def test_models(build_net, test_loader, device="cpu"):
    criterion = nn.CrossEntropyLoss()
    data = {}
    for model in (RUN_FOLDER / "bak").iterdir():
        print(model)
        tstamp = model.stem.split("_")[1]
        with torch_persistent_model(
            build_net,
            model,
            store_finally=False,
        ) as (model, _):
            
            
            lbl, pred,loss = test_model(
                model,  test_loader, n_batches=6, device=device
            )
            print(loss)
            data[tstamp] = loss

    ts = np.sort(list(data.keys()))
    ys = np.array([data[t] for t in ts])

    return ts, ys


def do_run_eval():
    model_transforms = EfficientNet_V2_S_Weights.DEFAULT.transforms()

    test_dataset = datasets.ImageFolder(
        root="test/",
        transform=model_transforms,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=True,
        pin_memory=True,
    )
    build_net = lambda *args, **kwargs: efficientnet_v2_s(
        *args, **kwargs, num_classes=len(test_dataset.classes)
    )

    test_data = test_models(build_net, test_loader)
    return test_data


def do_and_save_run_eval():

    test_t, test_y = do_run_eval()
    np.save(RUN_FOLDER / "test_t.npy", test_t)
    np.save(RUN_FOLDER / "test_y.npy", test_y)


def load_run_eval():

    test_t = np.load(RUN_FOLDER / "test_t.npy", allow_pickle=True)
    test_y = np.load(RUN_FOLDER / "test_y.npy", allow_pickle=True)

    return test_t, test_y


PROJ_FOLDER = Path(__file__).resolve().parent
RUN_FOLDER = PROJ_FOLDER / "runs" / "run_1"
BAKFOLDER = RUN_FOLDER / "bak"
LOGFOLDER = RUN_FOLDER / "logs"

if __name__ == "__main__":
    do_and_save_run_eval()
    test_t, test_y= load_run_eval()
    plt.plot(test_t, test_y)
    # plt.plot(run_t, run_y)
    plt.show()
