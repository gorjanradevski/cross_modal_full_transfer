import pytest
from utils.datasets import preprocess_caption, FlickrDataset


@pytest.fixture
def caption():
    return ".A man +-<      gOeS to BuY>!!!++-= BEER!?@#$%^& BUT BEER or BEER'S   "


@pytest.fixture
def caption_true():
    return "a man goes to buy beer but beer or beer's"


@pytest.fixture
def coco_id_to_captions_true():
    return {
        1: ["first caption", "fourth caption"],
        2: ["second caption", "fifth caption"],
        3: ["third caption"],
    }


@pytest.fixture
def flickr_texts_path():
    return "data/testing_assets/flickr_tokens.txt"


@pytest.fixture
def flickr_images_path():
    return "data/testing_assets/flickr_images/"


@pytest.fixture
def flickr_train_path():
    return "data/testing_assets/flickr_train.txt"


@pytest.fixture
def flickr_val_path():
    return "data/testing_assets/flickr_val.txt"


def test_preprocess_caption(caption, caption_true):
    caption_filtered = preprocess_caption(caption)
    print(caption_filtered)
    assert caption_filtered == caption_true


def test_flickr_parse_captions_filenames(flickr_texts_path):
    img_path_caption = FlickrDataset.parse_captions_filenames(flickr_texts_path)
    unique_img_paths = set()
    for img_path in img_path_caption.keys():
        assert len(img_path_caption[img_path]) == 5
        unique_img_paths.add(img_path)
    assert len(unique_img_paths) == 5
