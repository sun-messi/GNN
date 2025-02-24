# %%
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch.nn.init as init
import matplotlib.pyplot as plt

# %%
def generateAdj(N, p1, p2, c=100, sigma=0.005):
    # Create a matrix with random values from a normal distribution
    noise = torch.normal(mean=0, std=sigma, size=(N, N))

    # Create probability matrix
    prob_matrix = torch.full((N, N), p1)
    prob_matrix[c:, :] += (p2 - p1)
    prob_matrix[:, c:] += (p2 - p1)
    prob_matrix += noise
    prob_matrix = torch.clamp(prob_matrix, 0, 1)

    # Generate symmetric adjacency matrix
    mask = torch.triu(torch.ones(N, N), diagonal=1)
    adj_matrix = torch.bernoulli(prob_matrix) * mask
    symmetric_adj_matrix = adj_matrix + adj_matrix.T.clone()

    # Create edge_index from adjacency matrix
    edge_index = torch.nonzero(symmetric_adj_matrix, as_tuple=False).t()

    return edge_index

def generate_edge_index(N=2000, min_degree=1, max_degree=200, mean_degree=100, std_degree=20):
    # Generate normal random variables
    degrees = np.random.normal(loc=mean_degree, scale=std_degree, size=N).astype(int)
    
    # Truncate values to be within the desired range
    degrees = np.clip(degrees, min_degree, max_degree)

    rows = []
    cols = []

    for node, degree in enumerate(degrees):
        # Select 'degree' neighbors for the current node
        neighbors = np.random.choice(np.delete(np.arange(N), node), degree, replace=False)
        rows.extend([node] * degree)
        cols.extend(neighbors)

    # Create a tensor for row and column indices
    edge_index = torch.tensor([rows, cols], dtype=torch.int64)

    return edge_index

def generate_data(A_star, X, W, V, C, alpha):
    # A_star N*N, X N*d, W d*p, V K*p, C p*K,
    node_labels_F = torch.mm(A_star, torch.mm(X, W))
    node_labels_F = torch.mm(node_labels_F, C) # N*k
    node_labels_G_F_sin = torch.sin(torch.mm(A_star, torch.mm(node_labels_F, V)))
    node_labels_G_F_tanh = torch.tanh(node_labels_G_F_sin)
    node_labels_G_F_tanh = torch.tanh(torch.mm(A_star, torch.mm(node_labels_F, V)))
    node_labels_G_F = torch.mm(node_labels_G_F_sin * node_labels_G_F_tanh, C) # N*k
    # node_labels_G_F = torch.mm(node_labels_G_F_tanh, C) # N*k
    # node_labels_G_F = torch.matmul((torch.matmul(A_star, torch.matmul(node_labels_F, V))**4), C) #N*k
    node_labels_H = node_labels_F + alpha * node_labels_G_F
    return node_labels_H

def split_masks(num_nodes, train_rate, validate_rate, test_rate):
    assert train_rate + validate_rate + test_rate == 1.0, "Rates don't sum up to 1."

    # Create an array of zeros
    masks = np.zeros(num_nodes, dtype=np.bool)

    # Create node indices and shuffle them
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)

    # Set corresponding indices to True based on the rates
    train_end = int(num_nodes * train_rate)
    validate_end = train_end + int(num_nodes * validate_rate)

    masks[indices[:train_end]] = True
    train_mask = torch.tensor(masks)

    masks[:] = False
    masks[indices[train_end:validate_end]] = True
    validate_mask = torch.tensor(masks)

    masks[:] = False
    masks[indices[validate_end:]] = True
    test_mask = torch.tensor(masks)
    
    return train_mask, validate_mask, test_mask

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, A_star)
    loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, A_star)

    losses = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        loss = F.mse_loss(out[mask], data.y[mask])
        losses.append(float(loss.item()))

    return losses

def generate_normalized_adj_matrix(edge_index, num_nodes):
    # Create adjacency matrix
    adj_matrix = torch.sparse.FloatTensor(edge_index, torch.ones(edge_index.size(1)), torch.Size([num_nodes, num_nodes]))

    # Convert to dense for further operations
    adj_matrix = adj_matrix.to_dense()

    # Add self-loops
    adj_matrix += torch.eye(num_nodes)

    # Compute the degree matrix D
    degree_matrix = torch.diag(torch.sum(adj_matrix, dim=1))

    # Compute the inverse square root of the degree matrix
    D_inv_sqrt = torch.diag(1 / torch.sqrt(torch.diag(degree_matrix)))

    # Compute the normalized adjacency matrix
    normalized_adj_matrix = D_inv_sqrt @ adj_matrix @ D_inv_sqrt

    return normalized_adj_matrix

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_units, K):
        super(GCN, self).__init__()
        self.layer_1 = nn.Linear(in_channels, hidden_units)
        self.layer_2 = nn.Linear(hidden_units, hidden_units)
        self.out = nn.Linear(hidden_units, K)
        nn.init.normal_(self.out.weight, mean=0.0, std=1.0)

    def forward(self, x, normalized_adj_matrix):
        hidden_1 = F.relu(normalized_adj_matrix @ self.layer_1(x))
        hidden_2 = F.relu(normalized_adj_matrix @ self.layer_2(hidden_1))
        added_12 = hidden_1 + hidden_2
        logits = self.out(added_12)

        return logits

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# Constants
N = 2000
d = 100  # Feature dimension
m = 20  # Intermediate dimension
k = 5  # Number of output dimensions
alpha = 5  # Hyperparameter (adjust as needed)
train_rates = [0.2, 0.3, 0.4, 0.5, 0.6]
# configs = [(200, 200),(200, 180),(200, 150)]  # d1 and d2 configurations
configs = [200, 150, 100] 
hidden_channels = 50

# Number of times to repeat each experiment
num_runs = 5

final_test_losses_by_config = {}
for degree in configs:
    print(f"Running experiment with mean_degree={degree}")

    avg_final_test_losses = np.zeros(len(train_rates))
    one_norms = []
    for run in range(num_runs): # Repeat experiment
        print(f"    run={run}")
        # Generate graph
        edge_index = generate_edge_index(mean_degree=degree)
        
        # Randomly generate X, A_star, W, V, and C
        X = torch.randn(N, d)
        W = torch.randn(d, m)
        V = torch.randn(k, m)
        C = torch.randn(m, k)
        
        _, edge_weight = gcn_norm(edge_index) # Your output from gcn_norm

        # Create an empty matrix with the right size
        num_nodes = N
        A_star = generate_normalized_adj_matrix(edge_index, num_nodes)
        one_norm = torch.norm(A_star, p=1, dim=0).max()
        one_norms.append(one_norm)
        # Generate y using the previously defined function
        y = generate_data(A_star, X, W, V, C, alpha)

        # Create a PyTorch Geometric Data object
        data = Data(x=X, edge_index=edge_index, y=y)

        # Train and evaluate model
        final_test_losses = []
        for train_rate in train_rates:
            validate_rate = 0.2
            test_rate = 0.8 - train_rate
            data.train_mask, data.val_mask, data.test_mask = split_masks(num_nodes, train_rate, validate_rate, test_rate)

            data = data.to(device)
            A_star = A_star.to(device)

            model = GCN(X.shape[1], hidden_channels, y.shape[1]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
            
            min_val_loss = float('inf')
            best_test_loss = None
            train_losses, val_losses, test_losses = [], [], []

            for epoch in range(0, 3000):
                train_loss = train()
                train_loss, val_loss, test_loss = test()
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                test_losses.append(test_loss)

                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    best_test_loss = test_loss

                if epoch % 50 == 0:
                    lr = 1e-5

            final_test_losses.append(best_test_loss)

        # Add the current run's best test loss to the sum
        avg_final_test_losses += np.array(final_test_losses)

    # Divide by the number of runs to get the average
    avg_final_test_losses /= num_runs
    final_test_losses_by_config[(np.mean(one_norms))] = avg_final_test_losses.tolist()


np.save('figure_A_N/final_test_losses_by_config.npy', final_test_losses_by_config)
# Plot results
markers = ['o', 's', 'D'] # Markers for three lines
colors = ['b', 'g', 'r'] # Colors for three lines
linewidths = [5, 5, 5] # Linewidths for three lines
total_number_of_labeled_nodes = [rate * N for rate in train_rates]
plt.figure(figsize=(8, 6), dpi=300)
# Assuming `final_test_losses_by_config` is a dictionary with keys representing one_norm and values representing losses
for index, (one_norm, losses) in enumerate(final_test_losses_by_config.items()):
    plt.plot(total_number_of_labeled_nodes, np.log10(losses), marker=markers[index], markersize=12
             , color=colors[index], linewidth=linewidths[index], label = r'$||A^*||_1=$' + f'{one_norm:.2f}' )
plt.grid(which='both', linestyle='--', linewidth=0.5)
# plt.xscale('log')
plt.xlabel('Total number of labeled nodes',fontsize=14)
plt.ylabel('Test error',fontsize=14)
# plt.title('Final Test Loss vs Hidden Channels')
plt.legend(fontsize=10)
plt.xticks(fontsize=14) # Customize the x-axis tick labels
plt.yticks(fontsize=14) 
plt.show()
plt.savefig('figure_A_N/final_test_loss_plot.png', dpi=300)
