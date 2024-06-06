"""
Author:
-------
- Amos Matter (mail@amosmatter.ch)

License:
--------
- MIT License

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from eval_run import parse_logs


PROJ_FOLDER = Path(__file__).resolve().parent
RUN_FOLDER = PROJ_FOLDER / "runs" / "run_1"
LOG_FOLDER = RUN_FOLDER / "logs"


if __name__ == "__main__":
    fig = plt.figure()

    bigdf, f1 = parse_logs(LOG_FOLDER)
    y = bigdf["loss"] / 16
    loss_smooth = savgol_filter(y, len(bigdf) // 10, 3)
    ts = bigdf["timestamp"].to_numpy()
    ts = ts - ts[0]

    tstest = np.load(RUN_FOLDER / "test_t.npy").astype(float)
    ystest = np.load(RUN_FOLDER / "test_y.npy")

    print(tstest, ystest)
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    window_sz = len(bigdf) // 100

    # ax.scatter(bigdf["timestamp"], y, color="b", s=1)
    # ax.plot(bigdf["timestamp"], loss_smooth, color="r")
    ax1.semilogx(ts, loss_smooth, color="r")
    ax1.scatter(tstest - tstest[0], ystest, color="b", s=5)
    ax1.set_ylabel("loss")
    ax1.set_xlabel("training time (s)")
    # ax2.semilogx(tstest - tstest[0], ystest, color="r")
    ax2.semilogx(np.arange(len(f1)), f1)
    ax2.set_ylabel("f1 score")
    ax2.set_xlabel("percent of training done")
    # ax2 = ax1.twinx()
    # ax2.plot(bigdf["timestamp"], bigdf["epoch"], color="black")

    plt.show()
