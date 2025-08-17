import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

from data import generate_copy_data
from components.transformer import Transformer

def train(model,data,epochs=10,lr=1e-3,device="cpu"):
    optimizer = optim.Adam(model.parameters(),lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    model=model.to(device)
    losses=[]
    
    print("\n=====Training Started=====\n")
    for epoch in range(epochs):
        total_loss=0
        for source,target in data:
            source=torch.tensor([source],dtype=torch.long,device=device)
            target_in=torch.tensor([target[:-1]],dtype=torch.long,device=device)
            target_out=torch.tensor([target[1:]],dtype=torch.long,device=device)
            optimizer.zero_grad()
            output=model(source,target_in)
            loss=criterion(output.view(-1,output.size(-1)),target_out.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        avg_loss=total_loss/len(data)
        losses.append(avg_loss)
        print(f"Epoch[{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")
        
    print("\n=====Training Completed=====\n")
    return losses

def plot_losses(losses,save_path="training_loss.png"):
    plt.plot(losses,marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    
if __name__=="__main__":
    vocab_size=20
    seq_len=10
    num_samples=200
    epochs=15
    device="cuda" if torch.cuda.is_available() else "cpu"
    
    data=generate_copy_data(num_samples=num_samples,seq_len=seq_len,vocab_size=vocab_size)
    
    model = Transformer(
        source_vocab=vocab_size,
        target_vocab=vocab_size,
        d_model=64,
        n_heads=4,
        d_ff=128,
        n_layers=2,
        max_length=seq_len
    )
    
    losses=train(model,data,epochs=epochs,device=device)
    plot_losses(losses)
    
    os.makedirs("model_checkpoints",exist_ok=True)
    torch.save(model.state_dict(),"model_checkpoints/transformer_copy.pt")
    print("Model saved at model_checkpoints/transformer_copy.pt")