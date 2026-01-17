"""
Core evaluation logic for Referring Change Detection.
"""
import numpy as np
import torch
from tqdm import tqdm

from utils.metric import hist_info, compute_score
from utils.visualize import print_iou


def _is_main_process():
    """Check if we're on the main process (rank 0) for distributed runs."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def evaluate_referring_cd(
    dataset,
    network,
    llm_model,
    tokenizer,
    num_classes: int,
    device: torch.device,
    threshold: float = 0.5,
    lowercase_captions: bool = True,
    aggregation: str = "overwrite", 
    return_metrics: bool = False,
    show_progress: bool = True,
):
    """
    Evaluate a referring change detection model on a dataset.

    Args:
        dataset: ChangeDataset instance.
        network: The segmentation model (EncoderDecoder).
        llm_model: CLIP text encoder model.
        tokenizer: CLIP tokenizer.
        num_classes: Number of semantic classes.
        device: torch device.
        threshold: Sigmoid threshold for binary prediction.
        lowercase_captions: Whether to lowercase captions before encoding.
        aggregation: 'overwrite' (last class wins) or 'max_prob' (highest prob wins).
        return_metrics: If True, return a dict of all metrics.
        show_progress: Whether to show tqdm progress bar.

    Returns:
        If return_metrics=True: (hist, Score, metrics_dict)
        Else: (hist, Score)
    """
    network.eval()
    if llm_model is not None:
        llm_model.eval()

    is_main = _is_main_process()
    show_bar = show_progress and is_main

    hist = np.zeros((num_classes, num_classes), dtype=np.float64)
    total_correct = 0
    total_labeled = 0

    # Cache for text embeddings to avoid recomputing
    caption_cache = {}

    def get_caption_embedding(caption: str):
        """Get cached caption embedding."""
        key = caption.lower() if lowercase_captions else caption
        if key not in caption_cache:
            inputs = tokenizer(
                [key],
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                caption_cache[key] = llm_model(**inputs).last_hidden_state
        return caption_cache[key]

    iterator = tqdm(range(len(dataset)), desc="Evaluating", disable=not show_bar)

    for idx in iterator:
        data = dataset[idx]
        A = data["A"].unsqueeze(0).to(device)
        B = data["B"].unsqueeze(0).to(device)
        # gt is [1, H, W], squeeze to [H, W]
        label = data["gt"].squeeze(0).numpy()
        text_captions = data["text_captions"]
        class_indices = data["class_indexes"]

        H, W = label.shape

        # Initialize prediction
        if aggregation == "max_prob":
            pred_probs = torch.zeros((num_classes, H, W), device=device)
        pred = np.zeros_like(label, dtype=np.int32)

        # Process each class
        for class_idx, caption in zip(class_indices, text_captions):
            cap_str = str(caption).lower() if lowercase_captions else str(caption)
            cap_emb = get_caption_embedding(cap_str)

            with torch.no_grad():
                logits = network(A, B, captions=cap_emb)
                prob = logits.sigmoid().detach()[0, 0]  # [H, W]

            if aggregation == "max_prob":
                pred_probs[class_idx] = torch.maximum(pred_probs[class_idx], prob)
            else:  # overwrite
                mask = (prob > threshold).cpu().numpy()
                pred[mask] = class_idx

        # For max_prob, pick class with highest probability where prob > threshold
        if aggregation == "max_prob":
            max_probs, max_classes = pred_probs.max(dim=0)
            change_mask = (max_probs > threshold).cpu().numpy()
            pred[change_mask] = max_classes.cpu().numpy()[change_mask]

        # Compute metrics
        hist_tmp, labeled_tmp, correct_tmp = hist_info(num_classes, pred, label)
        hist += hist_tmp
        total_labeled += labeled_tmp
        total_correct += correct_tmp

    # Compute final metrics
    iou, Score, Sek, Semantic_IoU, semantic_f1, recall, precision, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(
        hist, total_correct, total_labeled
    )

    # Print results only on main process
    if is_main:
        class_names = getattr(dataset, "class_names", None)
        print_iou(
            iou, Score, Sek, Semantic_IoU, recall, precision,
            freq_IoU, mean_pixel_acc, pixel_acc, class_names
        )

    if return_metrics:
        metrics = {
            "Score": float(Score),
            "SeK": float(Sek),
            "Semantic_IoU": float(Semantic_IoU),
            "semantic_f1": float(semantic_f1),
            "recall": float(recall),
            "precision": float(precision),
            "freq_IoU": float(freq_IoU),
            "mean_pixel_acc": float(mean_pixel_acc),
            "pixel_acc": float(pixel_acc),
            "mean_IoU": float(np.nanmean(iou)),
        }
        # Add per-class IoU
        class_names = getattr(dataset, "class_names", None)
        for i, iou_val in enumerate(iou):
            name = class_names[i] if class_names else f"class_{i}"
            metrics[f"IoU_{name}"] = float(iou_val) if not np.isnan(iou_val) else 0.0
        return hist, Score, metrics

    return hist, Score
