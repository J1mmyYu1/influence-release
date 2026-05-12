#!/usr/bin/env python3
"""Minimal PyTorch influence functions demo on binary MNIST.

This script intentionally does not depend on TensorFlow. It trains a logistic
regression model on MNIST digits 0 vs 1, then computes:

1. Upweight effect for training points.
2. Leave-one-out approximation for removing a training point.
3. Helpful vs harmful training examples for one test example.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


@dataclass
class BinaryMnistData:
    train_x: Tensor
    train_y: Tensor
    test_x: Tensor
    test_y: Tensor


class LogisticRegression(nn.Module):
    """A single linear layer is logistic regression for binary classification."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PyTorch influence function demo for binary MNIST logistic regression."
    )
    parser.add_argument("--data-dir", default="data", help="Directory for MNIST data.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for saved results.")
    parser.add_argument("--digit-a", type=int, default=0, help="Negative class digit.")
    parser.add_argument("--digit-b", type=int, default=1, help="Positive class digit.")
    parser.add_argument("--train-size", type=int, default=1000, help="Number of train examples.")
    parser.add_argument("--test-size", type=int, default=200, help="Number of test examples.")
    parser.add_argument("--test-index", type=int, default=0, help="Test example to explain.")
    parser.add_argument("--top-k", type=int, default=5, help="How many helpful/harmful examples to print.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=0.5, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="L2 regularization strength.")
    parser.add_argument("--damping", type=float, default=1e-4, help="Extra Hessian damping.")
    parser.add_argument("--sanity-checks", type=int, default=3, help="Actual LOO retrains to compare.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def resolve_script_path(path: str) -> str:
    """Resolve relative paths next to this script so the demo stays self-contained."""
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(__file__), path)


def load_binary_mnist(args: argparse.Namespace, device: torch.device) -> BinaryMnistData:
    transform = transforms.ToTensor()
    data_dir = resolve_script_path(args.data_dir)

    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    def filter_digits(dataset: datasets.MNIST, limit: int) -> Tuple[Tensor, Tensor]:
        xs = dataset.data.float().div(255.0).view(-1, 28 * 28)
        labels = dataset.targets
        mask = (labels == args.digit_a) | (labels == args.digit_b)
        xs = xs[mask]
        labels = labels[mask]
        ys = (labels == args.digit_b).float()
        return xs[:limit].to(device), ys[:limit].to(device)

    train_x, train_y = filter_digits(train_set, args.train_size)
    test_x, test_y = filter_digits(test_set, args.test_size)
    return BinaryMnistData(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)


def train_model(
    train_x: Tensor,
    train_y: Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> LogisticRegression:
    torch.manual_seed(args.seed)
    model = LogisticRegression(train_x.shape[1]).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
    )

    for _ in range(args.epochs):
        for batch_x, batch_y in loader:
            logits = model(batch_x)
            data_loss = F.binary_cross_entropy_with_logits(logits, batch_y)
            # The Hessian below uses the same L2 term: (lambda / 2) * ||w||^2.
            reg_loss = 0.5 * args.weight_decay * model.linear.weight.pow(2).sum()
            loss = data_loss + reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def evaluate(model: LogisticRegression, x: Tensor, y: Tensor) -> Tuple[float, float]:
    with torch.no_grad():
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y).item()
        preds = (torch.sigmoid(logits) >= 0.5).float()
        acc = (preds == y).float().mean().item()
    return loss, acc


def per_example_weight_grads(model: LogisticRegression, x: Tensor, y: Tensor) -> Tensor:
    """Gradient of unregularized BCE loss with respect to the weight vector.

    For logistic regression, grad_i = (sigmoid(w^T x_i) - y_i) * x_i.
    """
    with torch.no_grad():
        probs = torch.sigmoid(model(x))
        return (probs - y).unsqueeze(1) * x


def logistic_hessian(model: LogisticRegression, x: Tensor, weight_decay: float, damping: float) -> Tensor:
    """Explicit Hessian of mean logistic loss plus L2 regularization."""
    with torch.no_grad():
        probs = torch.sigmoid(model(x))
        weights = probs * (1.0 - probs)
        weighted_x = x * weights.unsqueeze(1)
        hessian = x.T.matmul(weighted_x) / x.shape[0]
        eye = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
        return hessian + (weight_decay + damping) * eye


def upweight_effect(train_grad: Tensor, inverse_hvp: Tensor) -> Tensor:
    """Effect on test loss if each training point receives infinitesimal upweight.

    d loss_test / d epsilon_i = - grad_test^T H^{-1} grad_train_i.
    Negative means upweighting this training point would reduce the test loss.
    """
    return -train_grad.matmul(inverse_hvp)


def leave_one_out_approx(upweight_scores: Tensor, num_train: int) -> Tensor:
    """Approximate test-loss change after removing each training point.

    Removing one point is approximately the same as upweighting it by -1 / n.
    """
    return -upweight_scores / num_train


def helpful_harmful_indices(loo_scores: Tensor, top_k: int) -> Tuple[Tensor, Tensor]:
    """Helpful examples hurt the test point when removed; harmful examples help when removed."""
    helpful = torch.argsort(loo_scores, descending=True)[:top_k]
    harmful = torch.argsort(loo_scores, descending=False)[:top_k]
    return helpful, harmful


def print_examples(title: str, indices: Iterable[int], data: BinaryMnistData, up_scores: Tensor, loo_scores: Tensor) -> None:
    print(f"\n{title}")
    print(" rank | train_idx | label | upweight_effect | loo_loss_delta")
    print("------+-----------+-------+-----------------+---------------")
    for rank, idx_tensor in enumerate(indices, start=1):
        idx = int(idx_tensor)
        label = int(data.train_y[idx].item())
        print(
            f" {rank:>4} | {idx:>9} | {label:>5} | "
            f"{up_scores[idx].item():>15.8f} | {loo_scores[idx].item():>13.8f}"
        )


def retrain_without_one(
    data: BinaryMnistData,
    remove_idx: int,
    test_idx: int,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    keep_mask = torch.ones(data.train_x.shape[0], dtype=torch.bool, device=device)
    keep_mask[remove_idx] = False
    model = train_model(data.train_x[keep_mask], data.train_y[keep_mask], args, device)
    with torch.no_grad():
        test_logit = model(data.test_x[test_idx : test_idx + 1])
        test_loss = F.binary_cross_entropy_with_logits(test_logit, data.test_y[test_idx : test_idx + 1])
    return test_loss.item()


def sanity_check_retraining(
    data: BinaryMnistData,
    baseline_test_loss: float,
    loo_scores: Tensor,
    candidate_indices: Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    if args.sanity_checks <= 0:
        return

    print("\nLeave-one-out sanity check")
    print(" train_idx | approx_delta | actual_delta")
    print("-----------+--------------+-------------")
    for idx_tensor in candidate_indices[: args.sanity_checks]:
        idx = int(idx_tensor)
        actual_loss = retrain_without_one(data, idx, args.test_index, args, device)
        actual_delta = actual_loss - baseline_test_loss
        print(f" {idx:>9} | {loo_scores[idx].item():>12.8f} | {actual_delta:>11.8f}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_binary_mnist(args, device)
    if args.test_index >= data.test_x.shape[0]:
        raise ValueError(f"--test-index must be less than {data.test_x.shape[0]}")

    model = train_model(data.train_x, data.train_y, args, device)
    train_loss, train_acc = evaluate(model, data.train_x, data.train_y)
    test_loss, test_acc = evaluate(model, data.test_x, data.test_y)
    print(f"Device: {device}")
    print(f"Binary MNIST: {args.digit_a} vs {args.digit_b}")
    print(f"Train size: {data.train_x.shape[0]}, test size: {data.test_x.shape[0]}")
    print(f"Train loss/acc: {train_loss:.6f} / {train_acc:.4f}")
    print(f"Test loss/acc : {test_loss:.6f} / {test_acc:.4f}")

    test_x = data.test_x[args.test_index : args.test_index + 1]
    test_y = data.test_y[args.test_index : args.test_index + 1]
    with torch.no_grad():
        baseline_test_loss = F.binary_cross_entropy_with_logits(model(test_x), test_y).item()
    print(
        f"\nExplaining test index {args.test_index} "
        f"(label={int(test_y.item())}, loss={baseline_test_loss:.6f})"
    )

    train_grads = per_example_weight_grads(model, data.train_x, data.train_y)
    test_grad = per_example_weight_grads(model, test_x, test_y).squeeze(0)
    hessian = logistic_hessian(model, data.train_x, args.weight_decay, args.damping)
    inverse_hvp = torch.linalg.solve(hessian, test_grad)

    up_scores = upweight_effect(train_grads, inverse_hvp)
    loo_scores = leave_one_out_approx(up_scores, data.train_x.shape[0])
    helpful, harmful = helpful_harmful_indices(loo_scores, args.top_k)

    print_examples("Most helpful training examples", helpful, data, up_scores, loo_scores)
    print_examples("Most harmful training examples", harmful, data, up_scores, loo_scores)

    # Use the strongest examples for a small actual-retraining sanity check.
    sanity_candidates = torch.cat([helpful, harmful])
    sanity_check_retraining(data, baseline_test_loss, loo_scores, sanity_candidates, args, device)

    output_dir = resolve_script_path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "binary_mnist_influence_results.npz")
    np.savez(
        output_path,
        upweight_effect=up_scores.detach().cpu().numpy(),
        leave_one_out_delta=loo_scores.detach().cpu().numpy(),
        helpful_indices=helpful.detach().cpu().numpy(),
        harmful_indices=harmful.detach().cpu().numpy(),
        test_index=args.test_index,
        digit_a=args.digit_a,
        digit_b=args.digit_b,
    )
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
