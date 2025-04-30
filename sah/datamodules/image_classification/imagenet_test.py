import pytest

from sah.conftest import setup_with_overrides
from sah.datamodules.image_classification.image_classification_test import (
    ImageClassificationDataModuleTests,
)
from sah.datamodules.image_classification.imagenet import ImageNetDataModule
from sah.utils.testutils import needs_network_dataset_dir


@pytest.mark.slow
@needs_network_dataset_dir("imagenet")
@setup_with_overrides("datamodule=imagenet")
class TestImageNetDataModule(ImageClassificationDataModuleTests[ImageNetDataModule]): ...
