import torch
import torch.nn as nn
import math

try:
    from utils import download_and_load_model
except:
    from src.utils import download_and_load_model

class LoRA(nn.Module):
    def __init__(self, original_layer, r=4, alpha=32):
        """
        Low-Rank Adaptation (LoRA) module.
        
        Args:
            original_layer (nn.Module): The original layer to which LoRA is applied.
            r (int): Rank of the low-rank approximation.
            alpha (int): Scaling factor for the LoRA module.
        """
        super().__init__()
        # TODO: Initialize LoRA parameters
        self.r = r
        self.alpha = alpha
        self.original_layer = original_layer

        # TODO: Low-rank matrices A and B for LoRA
        self.A = torch.empty(self.original_layer.in_features, r)
        self.B = torch.zeros(r, self.original_layer.out_features)

        # TODO: Initialize LoRA weights (B is zero-initialized, A is random)
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        
        # TODO: Scaling factor alpha 
        self.scaling = alpha/r

        # TODO: Freeze the original layer parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False
                
    def forward(self, x):
        # TODO: Perform forward pass with low-rank update
        h = self.original_layer(x)
        lora_update = (x @ self.A @ self.B) * self.scaling
        return h + lora_update

def inject_lora_into_model(model, r=4, alpha=32, device='cpu'):
    """
    Inject LoRA layers into the linear layers of the attention modules of the model.
    
    Args:
        model (PreTrainedModel): The pre-trained model.
        r (int): Rank of the low-rank approximation.
        alpha (int): Scaling factor for LoRA.
        device (torch.device): The device to run the model on ('cuda' or 'cpu').
    
    Returns:
        model (PreTrainedModel): The model with LoRA injected into attention layers.
    """
    # TODO: Iterate through all child modules of the model
    child_list = ["encdecattention", "selfattention", "q", "k", "v", "o"]

    for child_name, child_module in model.named_modules():
        # TODO: Check if the child module is a linear layer of the attention module
        parts = child_name.split('.')
        last_part = parts[-1]
        if last_part in child_list:
            # TODO: Create LoRA layer for linear module
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)

            lora_layer = LoRA(child_module, r, alpha)
            setattr(parent, last_part, lora_layer)

    return model.to(device)


class SoftPromptEmbedding(nn.Module):
    def __init__(self, prompt_length, model_hidden_size):
        """
        Creates trainable soft prompts to prepend to input embeddings.

        Args:
            prompt_length (int): Number of virtual tokens in the soft prompt.
            model_hidden_size (int): The hidden size of the pre-trained model.
        """
        super().__init__()
        # TODO: Initialize soft prompt embeddings
        self.soft_prompt = nn.Parameter(torch.ones(prompt_length, model_hidden_size))

    def forward(self, input_embeddings):
        """
        Forward pass to prepend soft prompts to input embeddings.

        Args:
            input_embeddings (torch.Tensor): The original input embeddings from the tokenizer.

        Returns:
            torch.Tensor: The concatenated soft prompts and original embeddings.
        """
        # TODO: Expand soft prompt to match batch size
        batch_size = input_embeddings.size(0)
        soft_prompt_expanded = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)

        # TODO: Concatenate soft prompt and input embeddings
        return torch.cat((soft_prompt_expanded, input_embeddings), dim=1)