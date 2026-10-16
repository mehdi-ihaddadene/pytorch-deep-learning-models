--- FILE: .vscode.code-workspace ---
{
	"folders": [
		{
			"path": "../.vscode"
		},
		{
			"path": ".."
		},
		{
			"path": "../../../../../../wamp64/www"
		}
	],
	"settings": {
		"java.debug.settings.onBuildFailureProceed": true
	}
}

--- FILE: attention.py ---
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]  # key dimension
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)#dot product for each word with each other words creates a map matrix of each word that are imporant the the current word
    #math.sqrt(d_k) is scalling tool so that we dont get large numbers and the gradients wont be pushed to a place where become effectivily 0
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)#mask is for padding wish is dealing with empty padding and converting them to zero it also converts the future words so we can predict them

    weights = F.softmax(scores, dim=-1) #softmax sum up every row to 1 wish is percentages %
    return weights @ V, weights#this biscaly weight blend every word meaning in the sentence based on the calculated weigts 
 #multi head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):#the model inputs and spliting it to 8 heads 
        super().__init__()
        assert d_model % n_heads == 0 #makes sure that we split the heads evenly 
        self.d_k    = d_model // n_heads  # 64 each (512/8 = 64)
        self.n_heads = n_heads
        #learnable weight for each head in flexiable way that each one learn diffrent context of the word
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)#output weight it merges all the different head calculations and stitch them to one that rich in understanding
    def forward(self, x, mask=None):
        B, S, _ = x.shape#captures batch size sequence length and the input dimension _ is the place holder usualy 512 
        Q = self.Wq(x).view(B, S, self.n_heads, self.d_k).transpose(1,2)#the spilting wish is in this example from 512 to 8 heads 64 dimensions and transpose is for optimizations now its 8 batches instead of big one
        K = self.Wk(x).view(B, S, self.n_heads, self.d_k).transpose(1,2)
        V = self.Wv(x).view(B, S, self.n_heads, self.d_k).transpose(1,2)
        # Q,K,V: (B, n_heads, S, d_k)
        attn_out, _ = attention(Q, K, V, mask)#this is accensialy applying the attention function to each head calculation 
        attn_out = attn_out.transpose(1,2).reshape(B, S, -1)#attn_out.transpose(1, 2) it redo the swap we did earlier to put senquence length back and .reshape(B, S, -1) it glues back the 8 heads of 64 dimensions side-by-side back into the original 512-dimension size
        return self.Wo(attn_out)#processes the concatenated heads through a final linear layer to blend the 8 different perspectives into a unified output

--- FILE: CNN.py ---
import torch
import torch.nn as nn
import torch.nn.functional as F 
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
    
model = CNN()
x = torch.randn(8, 1, 28, 28)
out = model(x)
print(out.shape)
params= sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {params}")

--- FILE: CNNMNIST.py ---
import torch, torch.nn as nn, torch.optim as optim  # pytorch core modules
import torch.nn.functional as F  # functional operations like ReLU, softmax
from torch.utils.data import DataLoader  # batching and loading data efficiently
from torchvision import datasets, transforms, models  # pre-built datasets and models

device = "cuda" if torch.cuda.is_available() else "cpu"  # checks if GPU is available, uses it if yes otherwise falls back to CPU
test_tf = transforms.Compose([
    transforms.ToTensor(),  # converts PIL image to tensor with values 0-1
    transforms.Normalize([0.1307], [0.3081])  # subtract mean and divide by std for each channel to normalize (MNIST stats)
])
CLASSES = ["0","1","2","3","4","5","6","7","8","9"]  # digit labels 0-9

# ── 2. MODEL ──────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()  # class constructor that initializes the parent nn.Module class
        # first convolutional layer with stride to reduce spatial dimensions (height and width shrink)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)  # batch normalization to stabilize training and make it faster
        # second convolutional layer with stride 1 to keep spatial dimensions the same
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)  # another batch norm to stabilize the output
        # skip connection logic that makes the dimensions of input and output match for the residual addition
        self.skip  = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),  # 1x1 convolution to match dimensions
            nn.BatchNorm2d(out_ch)  # batch norm on the skip path
        ) if (in_ch != out_ch or stride != 1) else nn.Identity()  # if dimensions already match, do nothing (identity)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # conv -> batch norm -> relu activation
        out = self.bn2(self.conv2(out))  # conv -> batch norm (relu comes after skip addition)
        return F.relu(out + self.skip(x))  # add the skip connection and apply relu (this is the "residual" magic!)


class MNIST_ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # initial processing layer: 1 channel (grayscale) -> 64 channels to extract initial features
            nn.Conv2d(1, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            # stage 1: two residual blocks with 64 channels (no downsampling, keeps 28x28 size)
            ResBlock(64,  64),  ResBlock(64,  64),
            # stage 2: stride=2 shrinks spatial dims to 14x14, then normal block. goes from 64->128 channels
            ResBlock(64,  128, stride=2), ResBlock(128, 128),
            # stage 3: stride=2 shrinks spatial dims to 7x7, then normal block. goes from 128->256 channels
            ResBlock(128, 256, stride=2), ResBlock(256, 256),
            # adaptive average pooling collapses spatial dimensions from 7x7 to 1x1 (global average)
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),  # reshape from (batch, 256, 1, 1) to (batch, 256) for the linear layer
            # dropout to prevent overfitting by randomly zeroing 30% of activations during training
            nn.Dropout(0.3),
            # final classification layer: 256 features -> 10 digit classes (0-9)
            nn.Linear(256, 10)
        )

    def forward(self, x):
        # push input through the entire sequential model
        return self.net(x)

# ── 3. TRAINING FUNCTION ──────────────────────────

def train_epoch(model, loader, loss_fn, optimizer, scaler):
    model.train()  # switch model to training mode: enables dropout and uses batch norm running stats
    total_loss, correct = 0, 0  # initialize counters to track metrics over the entire epoch
    for X, y in loader:  # loop through batches of images (X) and labels (y)
        X, y = X.to(device), y.to(device)  # move data to GPU if available, otherwise stays on CPU
        # mixed precision: compute in lower precision (float16) to save memory and speed up training
        with torch.amp.autocast(device_type="cuda"):
            pred = model(X)  # forward pass: get predictions for the batch
            loss = loss_fn(pred, y)  # compute cross-entropy loss between predictions and true labels
        optimizer.zero_grad()  # clear old gradients from the last step (crucial!)
        # backward pass: compute gradients and scale them to prevent underflow in float16
        scaler.scale(loss).backward()
        scaler.step(optimizer)  # update model weights using the scaled gradients
        scaler.update()  # update the scaler for next iteration
        total_loss += loss.item()  # add this batch's loss to running total
        # count how many predictions match the true labels and add to correct counter
        correct    += (pred.argmax(1) == y).sum().item()
    # return average loss per batch and accuracy as percentage
    return total_loss / len(loader), correct / len(loader.dataset) * 100

def eval_epoch(model, loader, loss_fn):
    model.eval()  # switch model to evaluation mode: turns off dropout and uses batch norm learned stats (no updates)
    total_loss, correct = 0, 0  # initialize counters for loss and accuracy
    with torch.no_grad():  # disable gradient computation (we don't need gradients for evaluation, saves memory)
        for X, y in loader:  # loop through test/validation batches
            X, y = X.to(device), y.to(device)  # move data to device (GPU or CPU)
            pred = model(X)  # forward pass without tracking gradients
            # add this batch's loss to the running total
            total_loss += loss_fn(pred, y).item()
            # count correct predictions and add to counter
            correct    += (pred.argmax(1) == y).sum().item()
    # return average loss per batch and accuracy as percentage
    return total_loss / len(loader), correct / len(loader.dataset) * 100

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"  # redefine device just to be safe
    torch.manual_seed(42)  # set random seed for reproducibility (same results every run)

    # ── 1. DATA LOADER SETUP ─────────────────────
    # training augmentations to make model more robust and prevent overfitting
    train_tf = transforms.Compose([
        transforms.RandomRotation(10),  # rotate image slightly (±10 degrees) to handle rotated digits
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # randomly shift image position (±10%)
        transforms.ToTensor(),  # convert PIL image to tensor with values 0-1
        transforms.Normalize([0.1307], [0.3081])  # subtract mean and divide by std for MNIST (standardization)
    ])

    # load training data with augmentations
    train_data = datasets.MNIST("data", train=True,  download=True, transform=train_tf)
    # load test data without augmentations (we want to evaluate on original clean images)
    test_data  = datasets.MNIST("data", train=False, download=True, transform=test_tf)
    # create data loaders that batch and shuffle the data
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_data,  batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    # ── 4. TRAIN FROM SCRATCH ─────────────────────────
    model = MNIST_ResNet().to(device)  # create model and move it to device (GPU or CPU)
    # label smoothing makes the model less confident (outputs softer probabilities) which improves generalization
    loss_fn   = nn.CrossEntropyLoss(label_smoothing=0.1)
    # AdamW is an improved version of Adam with weight decay to prevent overfitting
    optimiz
... [Code snippet truncated locally to fit context ceilings] ...