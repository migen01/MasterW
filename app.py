import torch
import torch.nn as nn
import torch.optim as optim


class SimpleLNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    def logic_loss(self, output, target, logical_constraints):
        # Standard loss (e.g., Mean Squared Error)
        mse_loss = nn.MSELoss()(output, target)
    
        # Add logical constraint penalties
        penalty = 0
        for constraint in logical_constraints:
            penalty += constraint(output)
    
        total_loss = mse_loss + penalty
        return total_loss

# Logical constraint
def non_negative_constraint(output):
    return torch.sum(torch.clamp(-output, min=0))

# Define model parameters
input_size = 2
hidden_size = 5
output_size = 1

model = SimpleLNN(input_size, hidden_size, output_size)
# optimizer = optim.Adam(model.parameters(), lr=0.01)

# Define the logical constraints
logical_constraints = [non_negative_constraint]

# Example input and target data
input_data = torch.tensor([[0.5, -0.2], [1.5, 0.3], [-0.5, -1.2]], dtype=torch.float32)
target_data = torch.tensor([[1.0], [0.0], [1.0]], dtype=torch.float32)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    
    optimizer.zero_grad()
    
    output = model(input_data)
    
    # Call the logic_loss method of the model
    loss = model.logic_loss(output, target_data, logical_constraints)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Testing the model
output = model(input_data)
print("Model output after training:")
print(output)
