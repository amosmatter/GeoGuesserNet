"""
Author:
-------
- Amos Matter (mail@amosmatter.ch)

License:
--------
- MIT License

"""

from matplotlib import pyplot as plt
import torch
from torch import nn
from contextlib import contextmanager

import pandas as pd


@contextmanager
def torch_persistent_model(
    model_f, model_path, logger_path=None, logger_cols=None, store_finally=False,device="cpu"
):
    model = model_f()

    model_path.parent.mkdir(exist_ok=True, parents=True)

    if logger_path is not None and logger_cols is not None:
        logger_path.parent.mkdir(exist_ok=True, parents=True)

        try:
            logger = pd.read_csv(logger_path)
        except:
            logger = pd.DataFrame(columns=logger_cols)

    else:
        logger = None

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(e)
        print("Couldn't load model at", model_path)
        model = model_f()

    model = model.to(device)

    try:
        yield model, logger
    finally:
        if store_finally:
            logger.to_csv(logger_path, index=False)
            torch.save(model.state_dict(), model_path)
            print("Model saved to", model_path)

DEBUG = False
def eval_image(image, model, transform, chosen_classnames):

    image = transform(image)
    #image = image.to("cuda")
    image = image.unsqueeze(0)
    if DEBUG:
        plt.imshow(image.cpu().squeeze().permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.show()

    with torch.no_grad():
        model.eval()
        logits = model(image)
        softmax_out = nn.Softmax(dim=1)(logits)
    return {
        chosen_classnames[i]: softmax_out[0][i].item() for i in range(len(chosen_classnames))
    }
