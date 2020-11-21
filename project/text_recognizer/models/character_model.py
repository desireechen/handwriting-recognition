"""CharacterModel class."""
from typing import Callable, Dict, Tuple

import numpy as np

from text_recognizer.datasets.emnist_dataset import EmnistDataset  # This downloads the EMNIST dataset.
from text_recognizer.models.base import Model  # This is a Base class, to be subclassed by predictors for specific type of data. Model class, to be extended by specific types of models.
from text_recognizer.networks.mlp import mlp  # This creates a multi-layer perceptron. 


class CharacterModel(Model):
    """CharacterModel works on datasets providing images, with one-hot labels."""

    def __init__(
        self,
        dataset_cls: type = EmnistDataset,
        network_fn: Callable = mlp,
        dataset_args: Dict = None,
        network_args: Dict = None,
    ):
        super().__init__(dataset_cls, network_fn, dataset_args, network_args)

    def predict_on_image(self, image: np.ndarray) -> Tuple[str, float]:
        if image.dtype == np.uint8:
            image = (image / 255).astype(np.float32)
        # NOTE: integer to character mapping dictionary is self.data.mapping[integer]
        # Can amend below, if required. 
        pred_raw = self.network.predict(np.expand_dims(image, 0), batch_size=1).flatten()
        ind = np.argmax(pred_raw)
        confidence_of_prediction = pred_raw[ind]
        predicted_character = self.data.mapping[ind]

        return predicted_character, confidence_of_prediction