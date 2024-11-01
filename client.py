import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

random_seed=42
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
"""
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)        

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)  # Increase the number of neurons in the first hidden layer
        self.fc2 = nn.Linear(256, 128)      # Add a second hidden layer
        self.fc3 = nn.Linear(128, 10)       # Output layer for 10 classes
        self.dropout = nn.Dropout(0.3)  # 30% dropout to prevent overfitting

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)          # Apply dropout after the first hidden layer
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
      
class CNN_Cifar(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_Cifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 100)
        self.fc3 = nn.Linear(100, 84)
        self.fc4 = nn.Linear(84, 50)
        self.fc5 = nn.Linear(50, num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return self.logsoftmax(x)

"""
class MLP_FMNIST(nn.Module):
    def __init__(self, dim_in=784, dim_hidden1=64, dim_hidden2=30, dim_out=10):
        super(MLP_FMNIST, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden1 = nn.Linear(dim_hidden1, dim_hidden2)
        self.layer_hidden2 = nn.Linear(dim_hidden2, dim_out)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self.layer_input(x)
        x = self.relu(x)
        x = self.layer_hidden1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer_hidden2(x)
        return self.logsoftmax(x)   # Apply log-softmax and return

class Client:
    def __init__(self, client_id, cluster_id, data, model, device):
        self.client_id = client_id
        self.cluster_id = cluster_id
        self.data = data
        self.model = model.to(device)
        self.device = device
        self.gradients = None  # Store gradients here
        self.bandwidth = 1e6 #1 Mhz
        self.quantized_params = []
        self.power = 0.1 #0.1 joules per second
        self.metric = None

    def set_data(self, data):
        self.data = data
        
    def get_data(self):
        return self.data

    def set_model(self, global_model_state): #update model from global model
        self.model.load_state_dict(global_model_state)     
        
    def calculate_SNR(self):
        snr_db = np.random.uniform(0, 30)
        snr_linear = 10**(snr_db/10)
        return snr_linear
        

    def quantize_gradients(self, global_min_val, global_max_val, bits=2):
        print("\n\nInside quantize_gradients")
        
        min_val, max_val = global_min_val, global_max_val
        num_levels = 2 ** bits
        step_size = (max_val - min_val) / (num_levels - 1)

        self.quantized_gradients = []
        
        # Determine the appropriate integer type based on the number of bits
        if bits <= 7:
            dtype = torch.int8
        elif bits <= 15:
            dtype = torch.int16
        elif bits <= 31:
            dtype = torch.int32
        elif bits <= 63:
            dtype = torch.int64
        else:
            raise ValueError("Bits value too large, must be <= 63") 
        
        for grad in self.gradients:
            grad = grad.to(self.device)  # Move data to GPU
            quantized_grad = torch.round((grad - min_val) / step_size).to(dtype)
            self.quantized_gradients.append(quantized_grad)           
            
        self.dtype = dtype
        
        return self.quantized_gradients
     

    def find_min_max_grads(self):
        min_val = min(grad.min().item() for grad in self.gradients)
        max_val = max(grad.max().item() for grad in self.gradients)
        return min_val, max_val

    def pack_quantized_grads(self, bits):
        
        print("\n\nInside pack_quantized_grads")
        
        self.num_elements_in_Qgrads = sum(g.numel() for g in self.quantized_gradients)
        self.Qgrads_shapes = [grad.shape for grad in self.quantized_gradients]  # Store shapes
        
        # Flatten all quantized gradients into a single tensor
        flat_grads = torch.cat([grad.flatten() for grad in self.quantized_gradients]).int().cuda()

        # Calculate sign bits and absolute values
        sign_bits = (flat_grads < 0).int()
        abs_values = flat_grads.abs()
        
        # Combine sign bits with absolute values
        combined_values = (sign_bits << bits) | abs_values
        
        # Convert to binary representation, padded to fit the required bits
        num_bits_per_value = bits + 1
        binary_strings = combined_values.unsqueeze(-1).expand(-1, num_bits_per_value)
        binary_strings = ((combined_values.unsqueeze(-1) >> torch.arange(num_bits_per_value - 1, -1, -1, device=flat_grads.device)) & 1).view(-1)
        
        # Reshape binary strings into a byte-packed format
        total_bits = binary_strings.size(0)
        padded_size = (total_bits + 7) // 8 * 8
        binary_strings_padded = torch.nn.functional.pad(binary_strings, (0, padded_size - total_bits), "constant", 0)

        binary_strings_padded = binary_strings_padded.view(-1, 8)
        packed_bytes = torch.sum(binary_strings_padded * (2**torch.arange(7, -1, -1, device=flat_grads.device)), dim=1).byte()
        
        # Convert packed bytes to a bytearray
        self.packed_bytes = packed_bytes.cpu().numpy().tobytes() #The cpu() method is used to transfer tensors from GPU memory to CPU memory

        return self.packed_bytes
  
    
    def calculate_payload_size(self): #based on gradient
        payload_size_bits = 0
        for param in self.model.parameters():
            if param.grad is not None:
                num_elements = param.grad.numel()  # Number of elements at each param_grad in the gradient tensor
                element_size_bits = param.grad.element_size() * 8  # Size in bits (usually 32 or 64)
                payload_size_bits += num_elements * element_size_bits

        return payload_size_bits
    
    
    def calculate_transmission_time(self, payload_size_bits):
        
        snr = self.calculate_SNR()
        channel_capacity_bps = self.bandwidth * np.log2(1 + snr)
        
        self.transmitted_bits = payload_size_bits
        self.transmission_time = payload_size_bits / channel_capacity_bps
        
        return self.transmission_time
    
    def calculate_energy_consumption(self, transmission_time):
        # Implement a realistic model for transmission time calculation
        # This is a placeholder function
        self.energy = self.power * transmission_time  
        
    def get_num_samples(self):
            return len(self.data)
            
            
    def get_num_classes(self):
        """
        Returns the number of unique classes in the client's dataset.
        """
        class_counts = set()
        for _, target in self.data:
            class_counts.add(target)
            
        return len(class_counts)

    def train(self):
        self.model.train() #train mode
        data_loader = DataLoader(self.data, batch_size=64, shuffle=True)
        #data, target = next(iter(data_loader))
        
        for data, target in data_loader:
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            self.model.zero_grad()  # Clear previous gradients
            output = self.model(data)
            loss = torch.nn.functional.nll_loss(output, target)  # Compute the loss
            loss.backward()  # Compute gradients for model parameters
                   
            # Compute the Euclidean norm of all gradients as a single vector       
            self.local_l2_norm = torch.sqrt(sum(torch.sum(param.grad ** 2) for param in self.model.parameters())).item()
            self.local_loss = loss.item()
            self.gradients = [param.grad.clone() for param in self.model.parameters()]       

        return self.local_l2_norm

    # Function to evaluate the local accuracy of the model on client's local data
    def evaluate_local_accuracy(self):
        self.model.eval()  # Set model to evaluation mode
        data_loader = DataLoader(self.data, batch_size=64, shuffle=False)  # Use the local dataset
        correct = 0
        total = 0
        
        with torch.no_grad():  # Disable gradient calculation for evaluation
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)  # Get the predicted classes
                correct += pred.eq(target.view_as(pred)).sum().item()  # Count correct predictions
                total += target.size(0)  # Keep track of total samples
        
        accuracy = 100. * correct / total  # Calculate accuracy percentage
        return accuracy  # Return accuracy

class Cluster:
    def __init__(self, cluster_id, clients, K):
        self.cluster_id = cluster_id
        self.clients = clients
        self.K = K
        self.participating = False


def create_clusters(num_clusters, clients_per_cluster, K, device):
    clusters = []
    client_id = 0
    for cluster_id in range(num_clusters):
        clients = []
        for _ in range(clients_per_cluster[cluster_id]):
            model = MLP_FMNIST().to(device)
            clients.append(Client(client_id, cluster_id, None, model, device))
            client_id += 1
        clusters.append(Cluster(cluster_id, clients, K))
    return clusters
    
    
    
def verify_clusters(clusters, selected_clients):
        
        for cluster in clusters:
            cluster_selected_clients = [client for client,_,_ in selected_clients if client.cluster_id == cluster.cluster_id]
            if len(cluster_selected_clients) >= cluster.K:
                cluster.participating = True
            else:
                cluster.participating = False
                #print(f"Cluster {cluster.cluster_id} does not have enough valid clients.")
                #Need to create a way to prevent a client from participating if the malicious server calls it.



"""    
    def train_single_batch(self):
        self.model.train() #train mode
        data_loader = DataLoader(self.data, batch_size=128, shuffle=True)
        
        data, target = next(iter(data_loader))
        data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
        self.model.zero_grad()  # Clear previous gradients
        output = self.model(data)
        loss = torch.nn.functional.nll_loss(output, target)  # Compute the loss
        loss.backward()  # Compute gradients for model parameters
               
        # Compute the Euclidean norm of all gradients as a single vector       
        self.local_l2_norm = torch.sqrt(sum(torch.sum(param.grad ** 2) for param in self.model.parameters())).item()
        self.local_loss = loss.item()
        self.gradients = [param.grad.clone() for param in self.model.parameters()]       

        return self.local_l2_norm
"""