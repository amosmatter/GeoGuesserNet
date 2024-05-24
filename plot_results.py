import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


PROJ_FOLDER = Path(__file__).resolve().parent
LOG_FOLDER = PROJ_FOLDER / "runs" / "run_1" / "logs"


bigdf = pd.DataFrame()
for f in LOG_FOLDER.glob("*.csv"):
    print(f)
    df = pd.read_csv(f)
    bigdf = pd.concat([bigdf, df])

bigdf.sort_values(by="timestamp", inplace=True)
fig = plt.figure()

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


y = bigdf["loss"] / 16
loss_smooth = savgol_filter(y, len(bigdf) // 10, 3)
ts = bigdf["timestamp"].to_numpy()
ts = ts - ts[0] 
ax = fig.add_subplot(111)
# ax.scatter(bigdf["timestamp"], y, color="b", s=1)
# ax.plot(bigdf["timestamp"], loss_smooth, color="r")
ax.semilogx(ts[window_sz - 1 :: window_sz] , f1, color="r")

# ax2 = ax.twinx()
# ax2.plot(bigdf["timestamp"], bigdf["epoch"], color="black")


plt.show()
