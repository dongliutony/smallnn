"""Train the tiny GPT on a built-in corpus (CPU-friendly demonstration).

Run: python lessons/gpt_train.py
This is a language model, not an instruction-tuned chatbot.
"""
import torch

from gpt_model import MiniGPT

torch.manual_seed(42)
text = ("hello world. small models learn patterns.\n"
        "attention looks backward, never forward.\n"
        "we train by predicting the next character.\n") * 40
chars = sorted(set(text))
encode = {ch: i for i, ch in enumerate(chars)}
data = torch.tensor([encode[ch] for ch in text], dtype=torch.long)

context, batch_size = 16, 16
model = MiniGPT(vocab_size=len(chars), context=context)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

def get_batch():
    starts = torch.randint(len(data) - context - 1, (batch_size,))
    offsets = torch.arange(context)
    x = data[starts[:, None] + offsets]
    y = data[starts[:, None] + offsets + 1]
    return x, y

model.train()
for step in range(101):
    x, y = get_batch()
    _, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 25 == 0:
        print(f"step={step:03d} loss={loss.item():.3f}")

model.eval()
seed = torch.tensor([[encode["h"]]])
sample = model.generate(seed, steps=100)[0].tolist()
print("".join(chars[i] for i in sample))
