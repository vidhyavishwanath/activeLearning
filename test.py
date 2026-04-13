import torch
import torch.nn.functional as F


# ── Uncertainty helpers ───────────────────────────────────────────────────────

def max_confidence(probs: torch.Tensor) -> float:
    """Highest softmax probability — simple confidence score."""
    return probs.max().item()


def entropy_score(probs: torch.Tensor) -> float:
    """Normalised Shannon entropy (0 = certain, 1 = maximally uncertain)."""
    n_classes = probs.size(-1)
    entropy   = -(probs * probs.log().clamp(min=-1e9)).sum().item()
    return entropy / torch.log(torch.tensor(float(n_classes))).item()


CONCEPT_QUESTIONS = {
    0: "I am unsure here. Can you clarify what 'passive' behaviour looks like in this context?",
    1: "I am unsure here. Can you provide another example of 'active' behaviour?",
}

# ── Main test function ────────────────────────────────────────────────────────

def test_with_uncertainty(model, test_loader, criterion, device, confidence_threshold=0.75):
    """
    Evaluates model sample-by-sample.
    • Prints an uncertainty score for every sample.
    • If confidence < confidence_threshold the robot 'asks a question'
      (prints a clarification prompt) to flag it needs more information.
    """
    model.eval()

    total_loss  = 0.0
    correct     = 0
    n_queries   = 0          # how many times the robot asked for help
    dataset_len = len(test_loader.dataset)

    print(f"\n{'Sample':>7} | {'Pred':>4} | {'True':>4} | {'Conf':>6} | {'Entropy':>7} | Status")
    print("-" * 65)

    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)                          # (1, n_classes)
            loss   = criterion(output, target)
            total_loss += loss.item()

            probs      = F.softmax(output, dim=1).squeeze()   # (n_classes,)
            pred       = probs.argmax().item()
            confidence = max_confidence(probs)
            entropy    = entropy_score(probs)
            is_correct = (pred == target.item())
            correct   += int(is_correct)

            status = "✓" if is_correct else "✗"

            print(
                f"{idx+1:>7} | {pred:>4} | {target.item():>4} | "
                f"{confidence:>5.1%} | {entropy:>7.4f} | {status}"
            )

            # ── Active learning query ─────────────────────────────────────
            if confidence < confidence_threshold:
                n_queries += 1
                question = CONCEPT_QUESTIONS.get(
                    pred,
                    "I am uncertain — could you provide more context?"
                )
                print(f"          ↳ [QUERY #{n_queries}] Conf={confidence:.1%}  Entropy={entropy:.4f}")
                print(f"            ❓ {question}\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / dataset_len

    print("-" * 65)
    print(f"Test Loss : {avg_loss:.4f}")
    print(f"Accuracy  : {accuracy:.4f}  ({correct}/{dataset_len})")
    print(f"Queries   : {n_queries} sample(s) triggered a clarification question "
          f"({n_queries/dataset_len:.1%} of test set)")