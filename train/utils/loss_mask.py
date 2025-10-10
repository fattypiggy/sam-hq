import torch
from torch.nn import functional as F
from typing import List, Optional
import utils.misc as misc
from utils.skeleton_utils import batch_compute_tubed_skeleton

def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.
    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.
    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output

def cat(tensors: List[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

def get_uncertain_point_coords_with_randomness(
    coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio
):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.
    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.
    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    if num_random_points > 0:
        point_coords = cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))

def skeleton_recall_loss(
        src_masks: torch.Tensor,
        target_skeletons: torch.Tensor,
        num_masks: float,
        smooth: float = 1.0,
    ):
    """
    Compute Skeleton Recall Loss for connectivity preservation.

    Based on "Skeleton Recall Loss for Connectivity Conserving and Resource Efficient
    Segmentation of Thin Tubular Structures" (https://arxiv.org/abs/2404.03010)

    This loss computes soft recall of predictions on the tubed skeleton:
    L_SkelRecall = -1/|C| * Σ_c (Σ_i Y_skel_{i,c} * Ŷ_{i,c}) / (Σ_i Y_skel_{i,c})

    The loss encourages the network to include as much of the skeleton as possible
    in the prediction, thus preserving connectivity.

    Args:
        src_masks: Predicted masks (logits), shape (B, 1, H, W)
        target_skeletons: Tubed skeleton ground truth, shape (B, 1, H, W)
                         Values: 0 (background) or positive (skeleton regions)
        num_masks: Number of masks for normalization
        smooth: Smoothing factor to avoid division by zero (default: 1.0)

    Returns:
        Skeleton recall loss (scalar tensor)
    """
    # Apply sigmoid to get probabilities
    src_probs = src_masks.sigmoid()

    # Flatten spatial dimensions: (B, 1, H, W) -> (B, H*W)
    src_probs_flat = src_probs.flatten(1)
    target_skel_flat = target_skeletons.flatten(1)

    # Binarize skeleton (anything > 0 is skeleton)
    target_skel_binary = (target_skel_flat > 0).float()

    # Compute intersection: prediction overlap with skeleton
    intersection = (src_probs_flat * target_skel_binary).sum(-1)  # (B,)

    # Sum of skeleton pixels (denominator for recall)
    skeleton_sum = target_skel_binary.sum(-1)  # (B,)

    # Recall = intersection / skeleton_sum
    # Add smoothing to avoid division by zero
    recall = (intersection + smooth) / (skeleton_sum + smooth)

    # Return negative recall as loss (we want to maximize recall, minimize loss)
    loss = 1.0 - recall

    return loss.sum() / num_masks


skeleton_recall_loss_jit = torch.jit.script(
    skeleton_recall_loss
)  # type: torch.jit.ScriptModule


def loss_masks(src_masks, target_masks, num_masks, oversample_ratio=3.0):
    """Compute the losses related to the masks: the focal loss and the dice loss.
    targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
    """

    # No need to upsample predictions as we are using normalized coordinates :)

    with torch.no_grad():
        # sample point_coords
        point_coords = get_uncertain_point_coords_with_randomness(
            src_masks,
            lambda logits: calculate_uncertainty(logits),
            112 * 112,
            oversample_ratio,
            0.75,
        )
        # get gt labels
        point_labels = point_sample(
            target_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

    point_logits = point_sample(
        src_masks,
        point_coords,
        align_corners=False,
    ).squeeze(1)

    loss_mask = sigmoid_ce_loss_jit(point_logits, point_labels, num_masks)
    loss_dice = dice_loss_jit(point_logits, point_labels, num_masks)

    del src_masks
    del target_masks
    return loss_mask, loss_dice


def loss_masks_with_skeleton(
        src_masks: torch.Tensor,
        target_masks: torch.Tensor,
        target_skeletons: Optional[torch.Tensor],
        num_masks: float,
        oversample_ratio: float = 3.0,
        weight_skeleton: float = 1.0
    ):
    """
    Compute combined loss: focal + dice + skeleton recall.

    Args:
        src_masks: Predicted masks (logits), shape (B, 1, H, W)
        target_masks: Target masks, shape (B, 1, H, W)
        target_skeletons: Tubed skeletons, shape (B, 1, H, W). If None, no skeleton loss.
        num_masks: Number of masks for normalization
        oversample_ratio: Oversampling ratio for point sampling
        weight_skeleton: Weight for skeleton recall loss (default: 1.0)

    Returns:
        Tuple of (loss_mask, loss_dice, loss_skeleton) or (loss_mask, loss_dice, 0) if no skeleton
    """
    # Compute standard losses
    loss_mask, loss_dice = loss_masks(src_masks, target_masks, num_masks, oversample_ratio)

    # Compute skeleton loss if skeleton is provided
    if target_skeletons is not None and weight_skeleton > 0:
        # Resize skeleton to match src_masks spatial dimensions if needed
        if target_skeletons.shape[-2:] != src_masks.shape[-2:]:
            target_skeletons_resized = F.interpolate(
                target_skeletons,
                size=src_masks.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        else:
            target_skeletons_resized = target_skeletons

        loss_skeleton = skeleton_recall_loss_jit(src_masks, target_skeletons_resized, num_masks) * weight_skeleton
    else:
        loss_skeleton = torch.tensor(0.0, device=src_masks.device)

    return loss_mask, loss_dice, loss_skeleton



