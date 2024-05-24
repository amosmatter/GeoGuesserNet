# %%
from pathlib import Path
import numpy as np
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
from eval_functions import format_result,  test

# %%
N_EPOCHS = 50

# %%

transformss = EfficientNet_V2_S_Weights.DEFAULT.transforms()

# Load dataset from disk
train_dataset = datasets.ImageFolder(
    root="train/",
    transform=transformss,
)
test_dataset = datasets.ImageFolder(
    root="test/",
    transform=transformss,
)


# Define data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    pin_memory=True,
    pin_memory_device="cuda",
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=True,
    pin_memory=True,
    pin_memory_device="cuda",
)


def discard_pending_input():
    while msvcrt.kbhit():
        msvcrt.getch()


def build_net(*args, **kwargs):
    return efficientnet_v2_s(*args, **kwargs, num_classes=len(train_dataset.classes))


# %%
criterion = nn.CrossEntropyLoss()
run_folder = Path("runs/run_1")
log_folder = run_folder / "logs"
bak_folder = run_folder / "bak"
log_folder.mkdir(parents=True, exist_ok=True)
bak_folder.mkdir(parents=True, exist_ok=True)


with torch_persistent_model(
    build_net,
    run_folder / "model.pth",
    log_folder / "___effb0.csv",
    ["timestamp", "loss", "epoch", "labels", "preds"],
) as (
    model,
    logger,
):

    with open(run_folder / "labels.csv", "w") as f:
        f.write(",".join(train_dataset.classes))

    apparent_epoch_n = 1 + max(
        0,
        0,
        *(
            int(ffcomp)
            for file in log_folder.iterdir()
            if (ffcomp := file.stem.split("_")[0])
        ),
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lasts = []
    # Training loop
    for epoch in range(apparent_epoch_n, N_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        n_processed = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to("cuda"), labels.to("cuda")

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            step_loss = loss.item()
            n_processed += len(inputs)
            running_loss += step_loss
            lasts.append(step_loss)
            lbl: np.ndarray = labels.cpu().detach().numpy()
            outputs: np.ndarray = torch.argmax(outputs, dim=1).cpu().detach().numpy()
            if len(lasts) > 100:
                lasts.pop(0)

            print(
                format_result(train_dataset.classes, lbl, outputs) + 
                f"\nLoss: \tl1 : {step_loss:.4f}\tl100 : {sum(lasts) / len(lasts):.4f} \tProcessed {n_processed} / {len(train_dataset)}"
            )

            logger.loc[len(logger.index)] = [
                time.time(),
                step_loss,
                epoch,
                ";".join(str(num) for num in lbl),
                ";".join(str(num) for num in outputs),
            ]

            if keyboard.is_pressed("§"):
                discard_pending_input()
                logger.to_csv(
                    run_folder / f"logs/{epoch}_{time.time()}_effb0.csv",
                    index=False,
                )

                exit()
            if keyboard.is_pressed("space"):

                test(50, model, test_dataset, test_loader)
                model.train()
                discard_pending_input()

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch [{epoch}/{N_EPOCHS}], Loss: {epoch_loss:.4f}")
        epoch_logs = logger.loc[logger["epoch"] == epoch]
        epoch_logs.to_csv(
            run_folder / f"logs/{epoch}_{time.time()}_effb0.csv", index=False
        )

        torch.save(
            model.state_dict(),
            run_folder / f"bak/{epoch}_{time.time()}_{epoch_loss}_effb0.pth",
        )
