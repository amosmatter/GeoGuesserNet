"""
Author:
-------
- Amos Matter (mail@amosmatter.ch)

License:
--------
- MIT License

"""

# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import numpy as np
from torch import nn


GEO_CATEGORIES = {
    "NORTH_USA_CA": [
        "WashingtonDC",
        "Boston",
        "TRT",
        "Minneapolis",
        "Chicago",
    ],
    "SOUTH_USA_AU": [
        "Miami",
        "Phoenix",
        "Melbourne",
    ],
    "SE_ASIA": [
        "Bangkok",
        "Osaka",
    ],
    "MED_EUROPE": [
        "Madrid",
        "Barcelona",
        "Lisbon",
        "Rome",
    ],
    "TEMP_EUROPE": ["PRG", "PRS", "Brussels", "OSL", "London"],
}
INVERTED_CATEGORIES = {v: k for k, vs in GEO_CATEGORIES.items() for v in vs}


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


def test_model(model, test_loader, n_batches=-1, device="cuda"):
    model.eval()
    ctr = 0
    lbl, pred = torch.Tensor([]), torch.Tensor([])
    criterion = nn.CrossEntropyLoss()
    running_loss = 0
    n_processed = 0

    lbls_accum = np.array([])
    pred_accum = np.array([])

    with torch.no_grad():
        for inputs, labels in test_loader:
            print("run")
            if ctr >= n_batches and n_batches != -1:
                break
            ctr += 1

            inputs_batch, labels_batch = inputs.to(device), labels.to(device)

            outputs_batch = model(inputs_batch)
            loss = criterion(outputs_batch, labels_batch)
            step_loss = loss.item()

            n_processed += len(inputs)
            running_loss += step_loss
            print(step_loss)
            print(len(inputs))
            lbl: np.ndarray = labels_batch.cpu().detach().numpy()
            outputs: np.ndarray = torch.argmax(outputs_batch, dim=1).cpu().detach().numpy()

            lbls_accum = np.concatenate((lbls_accum, lbl))
            pred_accum = np.concatenate((pred_accum, outputs))

    epoch_loss = running_loss / n_processed

    return lbls_accum, pred_accum, epoch_loss
