import asyncio
from pathlib import Path
import sys
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
    QProgressBar,
)
from PySide6.QtGui import QPixmap,QImage
from PySide6.QtCore import Qt,QTimer,QThread
import numpy as np
import torch

from model import torch_persistent_model, eval_image
from torchvision.models.efficientnet import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.transforms import ToPILImage,ToTensor
from PIL import Image
from PIL import ImageGrab
import PySide6.QtAsyncio as QtAsyncio

def pil_image_to_qpixmap(pil_image: Image.Image | torch.Tensor) -> QPixmap:
    if isinstance(pil_image, torch.Tensor):
        pil_image =ToPILImage()(pil_image) 
    # Convert PIL image to RGBA if it is not already in this mode
    if pil_image.mode != "RGBA":
        pil_image = pil_image.convert("RGBA")

    # Get image data as bytes
    data = pil_image.tobytes("raw", "RGBA")

    # Create QImage from the bytes
    qimage = QImage(data, pil_image.width, pil_image.height, QImage.Format_RGBA8888)

    # Convert QImage to QPixmap
    qpixmap = QPixmap.fromImage(qimage)
    
    return qpixmap



class LiveImageApp(QMainWindow):
    def __init__(self, root, modell, categories, transforms):
        super().__init__()
        self.timer = QTimer(self)
        
        self.timer.timeout.connect(lambda: asyncio.ensure_future(self.on_image_selected()))
        self.timer.start(1000)
        
        self.modell = modell
        self.categories = categories
        self.transforms = transforms

        self.raw_image_label = QLabel()
        self.raw_image_label.setAlignment(Qt.AlignCenter)
        self.raw_image_label.setFixedSize(1200, 800)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(384, 384)
        
        self.rough_category_layout = QVBoxLayout()
        self.rough_category_layout.setAlignment(Qt.AlignTop)
        self.rough_category_widget = QWidget()
        self.rough_category_widget.setLayout(self.rough_category_layout)

        self.fine_category_layout = QVBoxLayout()
        self.fine_category_layout.setAlignment(Qt.AlignTop)
        self.fine_category_widget = QWidget()
        self.fine_category_widget.setLayout(self.fine_category_layout)


        self.store_rough_dict = {}
        self.store_rough_dict["labels"] = {}
        self.store_rough_dict["scores"] = {}

        self.store_fine_dict = {}
        self.store_fine_dict["labels"] = {}
        self.store_fine_dict["scores"] = {}
        
        central_widget = QWidget()
        central_layout = QHBoxLayout()
        #central_layout.addWidget(self.tree_view)
        central_layout.addWidget(self.raw_image_label)
        central_layout.addWidget(self.image_label)

        central_layout.addWidget(self.rough_category_widget)
        central_layout.addWidget(self.fine_category_widget)

        central_widget.setLayout(central_layout)
        self.setCentralWidget(central_widget)
        self.setWindowTitle("Image Selection App")
        self.setGeometry(100, 100, 800, 600)
        
    
    def display_results(self, scores, layout, store_dict):

        for category, score in scores.items():
            if category not in store_dict["labels"]:
                category_label = QLabel()
                category_label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
                layout.addWidget(category_label)
                progress_bar = QProgressBar()
                progress_bar.setRange(0, 100)
                progress_bar.setFixedWidth(100)
                layout.addWidget(progress_bar)

                store_dict["scores"][category] = progress_bar
                store_dict["labels"][category] = category_label

            store_dict["scores"][category].setValue(int(score * 100))
            store_dict["labels"][category].setText(
                f"{category} : {int(score * 10000) / 100:.2f}%"
            )        
    
    
     
    async def on_image_selected(self):
        screen = ImageGrab.grab()
        screen.convert("RGB")
        sz = screen.size
        height = sz[1]
        width = sz[0]
        
        vscale = 0.75
        hscale = 1
        rect = (width // 2 - width * hscale / 2, height // 2 -height * vscale / 2, width // 2 + width * hscale / 2, height // 2 + height * vscale / 2)
        screen = screen.crop(rect)        
        scores = eval_image(screen, self.modell, self.transforms, self.categories)
        raw_pixmap = pil_image_to_qpixmap(ToTensor()( screen) )
        processed_pixmap = pil_image_to_qpixmap( self.transforms( screen ) )
        
        self.raw_image_label.setPixmap(raw_pixmap.scaled(1200, 800, Qt.KeepAspectRatio))
        self.image_label.setPixmap(processed_pixmap.scaled(384, 384, Qt.KeepAspectRatio))

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

        result_coarse = {k: 0.0 for k in GEO_CATEGORIES.keys()}

        for region in GEO_CATEGORIES:
            for category in GEO_CATEGORIES[region]:
                result_coarse[region] += scores[category]
                
        self.display_results(result_coarse, self.rough_category_layout, self.store_rough_dict)
        self.display_results(scores, self.fine_category_layout, self.store_fine_dict)



if __name__ == "__main__":
    PROJ_FOLDER = Path(__file__).resolve().parent
    RUN_FOLDER = PROJ_FOLDER / "runs" / "run_1"
    app = QApplication(sys.argv)

    l8test_run = None
    l8est_i = 0
    for f in (RUN_FOLDER / "bak").iterdir():
        f_i = int(f.stem.split("_")[0])
        if f_i > l8est_i:
            l8test_run = f
            l8est_i = f_i

    categories = np.loadtxt(RUN_FOLDER / "labels.csv", dtype=str, delimiter=",")

    def build_net(*args, **kwargs):
        return efficientnet_v2_s(*args, **kwargs, num_classes=len(categories))

    transformss = EfficientNet_V2_S_Weights.DEFAULT.transforms()

    with torch_persistent_model(
        build_net,
        l8test_run,
        store_finally=False,
    ) as (modell, _):
        window = LiveImageApp(
            (PROJ_FOLDER / "val").as_posix(), modell, categories, transformss
        )
        window.show()
        QtAsyncio.run()
