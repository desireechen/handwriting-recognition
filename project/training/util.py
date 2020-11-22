"""Function to train a model."""
from time import time

from tensorflow.keras.callbacks import EarlyStopping, Callback

# Hide lines below until Weights & Biases is being used
import wandb
from wandb.keras import WandbCallback
# Hide lines above until Weights & Biases is being used

from text_recognizer.datasets.dataset import Dataset  # This is a simple abstract class for datasets. Dataset class to be extended by dataset-specific classes.
from text_recognizer.models.base import Model  # This is a Base class, to be subclassed by predictors for specific type of data. Model class to be extended by model-specific classes.

EARLY_STOPPING = True


# Hide lines below until Weights & Biases is being used
class WandbImageLogger(Callback):
    """Custom callback for logging image predictions"""

    def __init__(self, model_wrapper: Model, dataset: Dataset, example_count: int = 4):
        super().__init__()
        self.model_wrapper = model_wrapper
        self.val_images = dataset.x_test[:example_count]  # type: ignore

    def on_epoch_end(self, epoch, logs=None):
        images = [
            wandb.Image(image, caption="{}: {}".format(*self.model_wrapper.predict_on_image(image)))
            for i, image in enumerate(self.val_images)
        ]
        wandb.log({"examples": images}, commit=False)
# Hide lines above until Weights & Biases is being used


def train_model(model: Model, dataset: Dataset, epochs: int, batch_size: int, use_wandb: bool = False) -> Model:
    """Train model."""
    callbacks = []

    if EARLY_STOPPING:
        early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.01, patience=3, verbose=1, mode="auto")
        callbacks.append(early_stopping)

    # Hide lines below until Weights & Biases is being used
    if use_wandb:
        image_callback = WandbImageLogger(model, dataset)
        wandb_callback = WandbCallback()
        callbacks.append(image_callback)
        callbacks.append(wandb_callback)
    # Hide lines above until Weights & Biases is being used

    model.network.summary()

    t = time()
    _history = model.fit(dataset=dataset, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
    print("Training took {:2f} s".format(time() - t))

    return model
