# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import numpy as np


def format_result(classes, lbl, pred, k_width=20, d_width=32):
    outstr = "-" * 100 + "\n"

    outstr += f"{'CITY': >{k_width}} {'<+>:TRUE POS, <->:FALSE NEG, <x>:FALSE POS': ^{d_width *2}}\n"

    outstr += "-" * 100 + "\n"
    scale = d_width / (1.0 * len(lbl))
    tp_tot = 0
    fp_tot = 0
    for index, key in enumerate(classes):
        occ = np.sum((lbl == index), dtype=np.int64)
        tp = np.sum((lbl == index) & (pred == index), dtype=np.int64)
        fp = np.sum((lbl != index) & (pred == index), dtype=np.int64)
        tp_tot += tp
        fp_tot += fp
        prgstr = "-" * round(scale * (occ - tp)) + "+" * round(scale * tp)
        fstr = "x" * round(scale * fp)
        outstr += f"{key: >{k_width}} {prgstr: >{d_width}}|{fstr: <{d_width}}\n"

    prgstr = "+" * round(scale * tp_tot)
    fstr = "x" * round(scale * fp_tot)

    outstr += "-" * 100 + "\n"
    outstr += f"{'TOTAL': >{k_width}} {prgstr: >{d_width}}|{fstr: <{d_width}}\n\n"
    outstr += "-" * 100 + "\n"

    accuracy = accuracy_score(lbl, pred)
    precision = precision_score(lbl, pred, average="macro", zero_division=0)
    recall = recall_score(lbl, pred, average="macro", zero_division=0)
    f1 = f1_score(lbl, pred, average="macro", zero_division=0)

    outstr += f"""
Accuracy:   {accuracy:.3f}
Precision:  {precision:.3f}
Recall:     {recall:.3f}
F1:         {f1:.3f}
    """
    return outstr


def test(n_batches, model, test_dataset, test_loader):
    model.eval()
    print("=" * 100)

    print("Testing...")
    sums = {
        classname: {"false positive": 0, "true positive": 0, "occurrences": 0}
        for classname in test_dataset.classes
    }
    predictedaccum = np.array([])
    lblsaccum = np.array([])
    ctr = 0
    with torch.no_grad():

        for inputs, labels in test_loader:
            if ctr >= n_batches:
                break
            ctr += 1

            inputs, labels = inputs.to("cuda"), labels.to("cuda")

            outputs = model(inputs)

            lbl: np.ndarray = labels.cpu().detach().numpy()
            outputs: np.ndarray = torch.argmax(outputs, dim=1).cpu().detach().numpy()

            predictedaccum = np.concatenate((predictedaccum, outputs))
            lblsaccum = np.concatenate((lblsaccum, lbl))

    print(format_result(test_dataset.classes, lblsaccum, predictedaccum))

    print("=" * 100)
