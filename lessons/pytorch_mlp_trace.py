"""Trace one PyTorch MLP training step. Run: python pytorch_mlp_trace.py"""

import torch
from torch import nn


def main():
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 2))
    x = torch.randn(3, 4)       # batch=3, input features=4
    y = torch.tensor([0, 1, 0])  # one class index per example
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    print("1. FORWARD (batch of 3)")
    h_pre = model[0](x)
    h = model[1](h_pre)
    logits = model[2](h)
    loss = nn.functional.cross_entropy(logits, y)
    for name, tensor in [("x", x), ("h_pre", h_pre), ("h", h),
                         ("logits", logits), ("loss", loss)]:
        print(f"  {name:7s} shape={tuple(tensor.shape)!s:8s} "
              f"requires_grad={tensor.requires_grad}")

    print("\n2. BACKWARD (gradients, not updated weights)")
    optimizer.zero_grad()
    print("  before backward:", model[0].weight.grad)
    loss.backward()
    for name, p in model.named_parameters():
        print(f"  {name:8s} weight shape={tuple(p.shape)!s:8s} "
              f"grad shape={tuple(p.grad.shape)}")

    print("\n3. UPDATE")
    before = model[0].weight.detach().clone()
    optimizer.step()
    print("  first-layer weight changed:",
          not torch.equal(before, model[0].weight.detach()))

    print("\n4. INFERENCE (same weights, no autograd graph)")
    model.eval()
    with torch.no_grad():
        inference_logits = model(x)
    print("  logits shape:", tuple(inference_logits.shape))
    print("  requires_grad:", inference_logits.requires_grad)
    print("  grad_fn:", inference_logits.grad_fn)
    print("  parameter requires_grad remains:", model[0].weight.requires_grad)
    # no_grad() does not erase gradients previously stored in parameters.
    print("  previous parameter grad still exists:", model[0].weight.grad is not None)


if __name__ == "__main__":
    main()
