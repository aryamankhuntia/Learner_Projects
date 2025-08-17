import torch
from components.transformer import Transformer

def greedy_decode(model,source,max_length,start_symbol,device="cpu"):
    model.eval()
    source=source.to(device)
    source_mask=None
    
    memory=model.encoder(source,source_mask)
    ys=torch.ones(1,1).fill_(start_symbol).long().to(device)
    
    for i in range(max_length - 1):
        out=model.decoder(ys,memory,None,None)
        prob=out[:,-1]
        _,next_word=torch.max(prob,dim=1)
        next_word=next_word.item()
        
        ys=torch.cat([ys, torch.ones(1,1).type_as(source.data).fill_(next_word)],dim=1)
        
        if next_word==2:
            break
        
    return ys.squeeze().tolist()

if __name__ == "__main__":
    vocab_size = 20
    seq_len=10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Transformer(
        source_vocab=vocab_size,
        target_vocab=vocab_size,
        d_model=64,
        n_heads=4,
        d_ff=128,
        n_layers=2,
        max_length=seq_len,
    )
    model.load_state_dict(torch.load("model_checkpoints/transformer_copy.pt", map_location=device))
    model.to(device)

    print("\n===== Testing Model on Random Input =====\n")

    sample_source = torch.randint(3, vocab_size, (1, seq_len))
    prediction = greedy_decode(model, sample_source, max_length=seq_len, start_symbol=1, device=device)

    print("Source Sequence:   ", sample_source.squeeze().tolist())
    print("Predicted Output: ", prediction)
    print("\n===== Testing Completed =====\n")