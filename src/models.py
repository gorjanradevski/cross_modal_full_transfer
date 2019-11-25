import torch
from torch import nn
from torchvision.models import resnet152
from transformers import BertModel
import torch.nn.functional as F


class L2Normalize(nn.Module):
    def __init__(self):
        super(L2Normalize, self).__init__()

    def forward(self, x):
        norm = torch.pow(x, 2).sum(dim=1, keepdim=True).sqrt()
        normalized = torch.div(x, norm)

        return normalized


class ImageEncoder(nn.Module):
    def __init__(self, finetune: bool):
        super(ImageEncoder, self).__init__()
        self.resnet = torch.nn.Sequential(
            *(list(resnet152(pretrained=True).children())[:-1])
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        for param in self.resnet.parameters():
            param.requires_grad = finetune

    def forward(self, images: torch.Tensor):
        features = self.resnet(images)
        embedded_images = torch.flatten(features, start_dim=1)

        return embedded_images


class SentenceEncoder(nn.Module):
    def __init__(self, finetune: bool):
        super(SentenceEncoder, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        #  https://arxiv.org/abs/1801.06146

        for param in self.bert.parameters():
            param.requires_grad = finetune

    def forward(self, sentences: torch.Tensor):
        # TODO: Check about masking the padding
        # https://arxiv.org/abs/1801.06146
        hidden_states = self.bert(sentences)
        max_pooled = torch.max(hidden_states[0], dim=1)[0]
        mean_pooled = torch.mean(hidden_states[0], dim=1)
        # TODO: Check again about the CLS hidden state
        last_state = hidden_states[0][:, 0, :]
        embedded_sentences = torch.cat([last_state, max_pooled, mean_pooled], dim=1)

        return embedded_sentences


class Projector(nn.Module):
    def __init__(self, input_space, joint_space: int):
        super(Projector, self).__init__()
        self.fc1 = nn.Linear(input_space, joint_space)
        self.bn = nn.BatchNorm1d(joint_space)
        self.fc2 = nn.Linear(joint_space, joint_space)
        self.l2_normalize = L2Normalize()

    def forward(self, embeddings: torch.Tensor):
        projected_embeddings = self.fc2(self.bn(F.relu(self.fc1(embeddings))))

        return self.l2_normalize(projected_embeddings)


class ImageTextMatchingModel(nn.Module):
    def __init__(self, joint_space: int, finetune: bool = False):
        super(ImageTextMatchingModel, self).__init__()
        self.finetune = finetune
        # Image encoder
        self.image_encoder = ImageEncoder(finetune)
        self.image_encoder.eval()
        self.image_projector = Projector(2048, joint_space)
        # Sentence encoder
        self.sentence_encoder = SentenceEncoder(finetune)
        self.sentence_encoder.eval()
        self.sentence_projector = Projector(768 * 3, joint_space)

    def forward(self, images: torch.Tensor, sentences: torch.Tensor):
        embedded_images = self.image_encoder(images)
        embedded_sentences = self.sentence_encoder(sentences)

        return (
            self.image_projector(embedded_images),
            self.sentence_projector(embedded_sentences),
        )

    def train(self, mode: bool = True):
        if self.finetune and mode:
            self.image_encoder.train()
            self.sentence_encoder.train()
            self.image_projector.train(True)
            self.sentence_projector.train(True)
        elif mode:
            self.image_projector.train(True)
            self.sentence_projector.train(True)
        else:
            self.image_encoder.train(False)
            self.sentence_encoder.train(False)
            self.image_projector.train(False)
            self.sentence_projector.train(False)


class TripletLoss(nn.Module):
    def __init__(self, margin: float, device: str):
        """Build the batch-hard triplet loss over a batch of embeddings.

        Arguments:
            margin: margin for triplet loss.
            device: on which device to compute the loss.
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.device = device

    def forward(
        self,
        labels: torch.Tensor,
        image_embeddings: torch.Tensor,
        sentence_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the triplet loss using batch-hard mining.

        Arguments:
            labels: The labels for each of the embeddings.
            embeddings: The embeddings.

        Returns:
            Scalar loss containing the triplet loss.
        """
        # Get the distances
        pairwise_dist = torch.matmul(image_embeddings, sentence_embeddings.t())

        # For each anchor, get the hardest positive
        # First, we need to get a mask for every valid positive (they should have same
        # label)
        mask_anchor_positive = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        # We put to 0 any element where (a, p) is not valid (valid if a != p and
        # label(a) == label(p))
        anchor_positive_dist = mask_anchor_positive * pairwise_dist

        # shape (batch_size, 1)
        hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

        # For each anchor, get the hardest negative
        # First, we need to get a mask for every valid negative (they should have
        # different labels)
        mask_anchor_negative = (~(labels.unsqueeze(0) == labels.unsqueeze(1))).float()

        # We add the maximum value in each row to the invalid negatives
        # (label(a) == label(n))
        max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (
            1.0 - mask_anchor_negative
        )

        # shape (batch_size,)
        hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        tl = hardest_positive_dist - hardest_negative_dist + self.margin
        tl[tl < 0] = 0
        triplet_loss = tl.mean()

        return triplet_loss


class BatchAll(nn.Module):
    def __init__(self, margin: float, device: str):
        """Build the batch-all triplet loss over a batch of embeddings.

        Arguments:
            margin: margin for triplet loss
            device: on which device to compute the loss.
        """
        super(BatchAll, self).__init__()
        self.margin = margin
        self.device = device

    def forward(
        self, labels: torch.Tensor, image_embeddings: torch.Tensor, sentence_embeddings
    ) -> torch.Tensor:
        """Computes the triplet loss using batch-hard mining.

        Arguments:
            labels: The labels for each of the embeddings.
            embeddings: The embeddings.

        Returns:
            Scalar loss containing the triplet loss.

        """
        # Get the distance matrix
        pairwise_dist = torch.matmul(image_embeddings, sentence_embeddings.t())

        anchor_positive_dist = pairwise_dist.unsqueeze(2)
        anchor_negative_dist = pairwise_dist.unsqueeze(1)

        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j,
        # negative=k
        # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
        # and the 2nd (batch_size, 1, batch_size)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin

        # Put to zero the invalid triplets
        # (where label(a) != label(p) or label(n) == label(a) or a == p)
        mask = _get_triplet_mask(labels)
        triplet_loss = mask.float() * triplet_loss

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss[triplet_loss < 0] = 0

        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = triplet_loss[triplet_loss > 1e-16]
        num_positive_triplets = valid_triplets.size(0)

        # Get final mean triplet loss over the positive valid triplets
        triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

        return triplet_loss


def _get_triplet_mask(labels: torch.Tensor) -> torch.Tensor:
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:

        - labels[i] == labels[j] and labels[i] != labels[k]

    Args:
        labels: torch.int32 `Tensor` with shape [batch_size].

    Returns:
        The triplet mask.

    """
    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)

    valid_labels = ~i_equal_k & i_equal_j

    return valid_labels
