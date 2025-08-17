import torch

def make_source_mask(source):
        return (source!=0).unsqueeze(1).unsqueeze(2)
    
def make_target_mask(target):
    batch_size,sequence_length=target.shape
    mask=torch.tril(torch.ones(sequence_length,sequence_length)).unsqueeze(0).unsqueeze(0)
                                                                            
    return mask.to(target.device)