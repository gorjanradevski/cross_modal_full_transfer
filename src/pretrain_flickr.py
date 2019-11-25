import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

from utils.datasets import FlickrDatasetTrain, FlickrDatasetValTest
from utils.evaluator import Evaluator
from utils.data_loading_utils import collate_pad_batch
from models import ImageTextMatchingModel, TripletLoss


def train(
    images_path: str,
    sentences_path: str,
    train_imgs_file_path: str,
    val_imgs_file_path: str,
    epochs: int,
    batch_size: int,
    save_model_path: str,
    learning_rate: float,
    weight_decay: float,
    clip_val: float,
    joint_space: int,
    margin: float,
):
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_train = FlickrDatasetTrain(
        images_path, sentences_path, train_imgs_file_path
    )
    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_pad_batch,
        pin_memory=True,
    )
    dataset_val = FlickrDatasetValTest(images_path, sentences_path, val_imgs_file_path)
    val_loader = DataLoader(
        dataset_val,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_pad_batch,
        pin_memory=True,
    )
    model = nn.DataParallel(ImageTextMatchingModel(joint_space, finetune=False)).to(
        device
    )
    criterion = TripletLoss(margin, device)
    # noinspection PyUnresolvedReferences
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    evaluator = Evaluator(len(dataset_val), joint_space)
    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}...")
        evaluator.reset_all_vars()

        # Set model in train mode
        model.train(True)
        # remove past gradients
        optimizer.zero_grad()
        with tqdm(total=len(train_loader)) as pbar:
            for images, sentences in train_loader:
                images, sentences = images.to(device), sentences.to(device)
                # forward
                embedded_images, embedded_sentences = model(images, sentences)
                loss = criterion(
                    torch.arange(len(sentences)).to(device),
                    embedded_images,
                    embedded_sentences,
                )
                # backward
                loss.backward()
                # clip the gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                # update weights
                optimizer.step()
                # Remove gradients
                optimizer.zero_grad()
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({"Batch loss": loss.item()})

        # Set model in evaluation mode
        model.train(False)
        with torch.no_grad():
            for images, sentences in tqdm(val_loader):
                images, sentences = images.to(device), sentences.to(device)
                embedded_images, embedded_sentences = model(images, sentences)
                evaluator.update_embeddings(
                    embedded_images.cpu().numpy().copy(),
                    embedded_sentences.cpu().numpy().copy(),
                )

        if evaluator.is_best_recall_at_k():
            evaluator.update_best_recall_at_k()
            print("=============================")
            print(
                f"Found new best on epoch {epoch + 1}!! Saving model!\n"
                f"Current image-text recall at 1, 5, 10: "
                f"{evaluator.best_image2text_recall_at_k} \n"
                f"Current text-image recall at 1, 5, 10: "
                f"{evaluator.best_text2image_recall_at_k}"
            )
            print("=============================")
            torch.save(model.state_dict(), save_model_path)
        else:
            print("=============================")
            print(
                f"Metrics on epoch {epoch + 1}\n"
                f"Current image-text recall at 1, 5, 10: "
                f"{evaluator.cur_image2text_recall_at_k} \n"
                f"Current text-image recall at 1, 5, 10:"
                f"{evaluator.cur_text2image_recall_at_k}"
            )
            print("=============================")


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    train(
        args.images_path,
        args.texts_path,
        args.train_imgs_file_path,
        args.val_imgs_file_path,
        args.epochs,
        args.batch_size,
        args.save_model_path,
        args.learning_rate,
        args.weight_decay,
        args.clip_val,
        args.joint_space,
        args.margin,
    )


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(
        description="Trains a model on the Flickr8k or Flickr30k datasets."
    )
    parser.add_argument(
        "--images_path",
        type=str,
        default="data/Flickr8k_dataset/Flickr8k_Dataset",
        help="Path where all images are.",
    )
    parser.add_argument(
        "--texts_path",
        type=str,
        default="data/Flickr8k_dataset/Flickr8k_text/Flickr8k.token.txt",
        help="Path to the file where the image to caption mappings are.",
    )
    parser.add_argument(
        "--train_imgs_file_path",
        type=str,
        default="data/Flickr8k_dataset/Flickr8k_text/Flickr_8k.trainImages.txt",
        help="Path to the file where the train images names are included.",
    )
    parser.add_argument(
        "--val_imgs_file_path",
        type=str,
        default="data/Flickr8k_dataset/Flickr8k_text/Flickr_8k.devImages.txt",
        help="Path to the file where the validation images names are included.",
    )
    parser.add_argument(
        "--save_model_path",
        type=str,
        default="models/pretrained_flickr8k.pt",
        help="Where to save the model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="The number of epochs to train the model excluding the vgg.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="The size of the batch."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0002, help="The learning rate."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="The weight decay."
    )
    parser.add_argument(
        "--joint_space",
        type=int,
        default=512,
        help="The joint space where the encodings will be projected.",
    )
    parser.add_argument(
        "--margin", type=float, default=0.2, help="The contrastive margin."
    )
    parser.add_argument(
        "--clip_val", type=float, default=2.0, help="The clipping threshold."
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
