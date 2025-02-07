import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Define the LongTermMemory module
class LongTermMemory(nn.Module):
    def __init__(self, input_dim, memory_dim, learn_rate=0.1):
        super(LongTermMemory, self).__init__()
        # Initialize memory as a learnable parameter of shape [memory_dim, input_dim]
        self.memory = nn.Parameter(torch.zeros(memory_dim, input_dim))
        self.learn_rate = learn_rate

    def forward(self, x):
        """
        x: Tensor of shape [batch, seq_length, input_dim]
        We compute the difference between every token and each memory slot,
        then compute a 'surprise' value and aggregate updates over batch and sequence.
        """
        # Expand dimensions to perform pairwise subtraction:
        # self.memory: [memory_dim, input_dim] -> [1, 1, memory_dim, input_dim]
        mem_exp = self.memory.unsqueeze(0).unsqueeze(0)
        # x: [batch, seq_length, input_dim] -> [batch, seq_length, 1, input_dim]
        x_exp = x.unsqueeze(2)
        # Compute difference: shape becomes [batch, seq_length, memory_dim, input_dim]
        diff = x_exp - mem_exp
        # Compute 'surprise' as the norm along the feature dimension: [batch, seq_length, memory_dim, 1]
        surprise = torch.norm(diff, dim=-1, keepdim=True)
        # Compute update: [batch, seq_length, memory_dim, input_dim]
        update = self.learn_rate * surprise * diff
        # Aggregate the updates over the batch and sequence dimensions:
        aggregated_update = update.mean(dim=(0, 1))  # shape: [memory_dim, input_dim]
        # Update memory (in-place update)
        self.memory.data += aggregated_update
        return self.memory

# Define the Titans model that uses the long-term memory module and an attention mechanism
class Titans(nn.Module):
    def __init__(self, input_dim, hidden_dim, memory_dim):
        super(Titans, self).__init__()
        self.memory_module = LongTermMemory(input_dim, memory_dim)
        # MultiheadAttention expects inputs with shape [batch, seq_length, embed_dim] when batch_first=True
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4, batch_first=True)
        self.fc = nn.Linear(input_dim, hidden_dim)
    
    def forward(self, x):
        """
        x: Tensor of our shape [batch, seq_length, input_dim]
        The memory module updates its internal state based on x.
        Then we use the (updated) memory as key/value for the attention module.
        """
        # Update memory (the return value is not used further)
        _ = self.memory_module(x)
        # Expand memory to include the batch dimension: [batch, memory_dim, input_dim]
        mem = self.memory_module.memory.unsqueeze(0).expand(x.size(0), -1, -1)
        # Use x as the query and memory as key/value in the attention module
        attn_out, _ = self.attention(query=x, key=mem, value=mem)
        # Pass the attention output through a fully connected layer with ReLU activation
        return F.relu(self.fc(attn_out))

# Function to generate a synthetic input sequence
def generate_sequence(seq_length=50, input_dim=128, batch_size=1):
    # Returns a tensor of shape [batch_size, seq_length, input_dim]
    return torch.randn(batch_size, seq_length, input_dim)

# Function to run the model
def run_model():
    model = Titans(input_dim=128, hidden_dim=256, memory_dim=10)
    input_seq = generate_sequence()  # default: batch_size=1, seq_length=50, input_dim=128
    output = model(input_seq)
    return model, output

# Function to visualize the memory activations
def visualize_memory(model):
    memory_updates = model.memory_module.memory.detach().cpu().numpy()
    plt.imshow(memory_updates, aspect='auto', cmap='viridis')
    plt.colorbar(label='Memory Activation')
    plt.title('Titans Memory Module Activation')
    plt.xlabel('Feature Dimensions')
    plt.ylabel('Memory Slots')
    plt.show()

if __name__ == '__main__':
    model, output = run_model()
    visualize_memory(model)

