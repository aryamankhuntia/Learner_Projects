import torch
import torch.nn as nn

def train(model, data, epochs=10, lr=1e-3, device="cpu"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model = model.to(device)
    losses = []

    for epoch in range(epochs):
        total_loss = 0
        for src, tgt in data:
            src = torch.tensor([src], dtype=torch.long, device=device)
            tgt_in = torch.tensor([tgt[:-1]], dtype=torch.long, device=device)
            tgt_out = torch.tensor([tgt[1:]], dtype=torch.long, device=device)

            optimizer.zero_grad()
            output = model(src, tgt_in)

            loss = criterion(output.view(-1, output.size(-1)), tgt_out.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss = {avg_loss:.4f}")
    return losses
