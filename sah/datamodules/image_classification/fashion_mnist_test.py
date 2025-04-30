from sah.conftest import setup_with_overrides
from sah.datamodules.image_classification.fashion_mnist import FashionMNISTDataModule
from sah.datamodules.image_classification.image_classification_test import (
    ImageClassificationDataModuleTests,
)


@setup_with_overrides("datamodule=fashion_mnist")
class TestFashionMNISTDataModule(ImageClassificationDataModuleTests[FashionMNISTDataModule]): ...
