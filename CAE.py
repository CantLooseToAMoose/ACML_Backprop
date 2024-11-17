import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset

from torch.utils.data import Dataset
from torch.utils.data import random_split

torch.manual_seed(42)

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

'''Dataset for our standard rgb to rgb autoencoder: returns images as both inputs and targets'''
class AutoencoderDataset(Dataset):
    def __init__(self, original_dataset):
        self.dataset = original_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]  
        return image, image  

'''Dataset for our grayscale to rgb autoencoder: inputs are grayscale images, targets are color images'''
class ColorEncoderDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.to_grayscale = transforms.Grayscale(num_output_channels=1) 

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        color_image, _ = self.dataset[idx]  
        grayscale_image = self.to_grayscale(transforms.ToPILImage()(color_image))  
        return transforms.ToTensor()(grayscale_image), color_image  

'''Dataset for our chrominance autoencoder: inputs are luminance and targets are chrominance'''
class ChrominanceEncoderDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        color_image, _ = self.dataset[idx] 
        color_image_pil = transforms.ToPILImage()(color_image)  

        yuv_image = color_image_pil.convert("YCbCr")
        y, u, v = yuv_image.split()  

        # Convert to tensors and normalize UV channels to [-1, 1]
        y_tensor = transforms.ToTensor()(y)  # Grayscale (Luminance)
        uv_tensor = torch.stack([transforms.ToTensor()(u), transforms.ToTensor()(v)], dim=0)
        uv_tensor = uv_tensor.squeeze()  # Chrominance with 2 channels (UV)
        uv_tensor = (uv_tensor - 0.5) * 2.0  

        return y_tensor, uv_tensor  





import torch.nn as nn

import torch.optim as optim

'''Simple Convolutional Autoencoder with symmetric encoder and decoder'''
# Define a simple CNN model
class SimpleCAE(nn.Module):
    def __init__(self):
        super(SimpleCAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, padding=0),
            nn.Conv2d(8, 12, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, padding=0),
            nn.Conv2d(12, 16, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(16, 12, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(12, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print(f"Received input Data with shape: {x.shape}")
        x = self.encoder(x)
        # print(f"Encoder finished and has Latent Space Representation of shape: {x.shape}")
        x = self.decoder(x)
        # print(f"Decoder finished and output has shape: {x.shape}")
        return x

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


class CAE_1(nn.Module):
    """
    Same Latent space but more informed Dimensionality Reduction and Increase
    """
    def __init__(self):
        super(CAE_1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 12, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(12, 16, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 12, 3, padding=1,stride=2, output_padding=1),  # Doubles spatial dimensions
            nn.ReLU(),
            nn.ConvTranspose2d(12, 8, 3, padding=1,stride=2, output_padding=1),  
            nn.ReLU(),
            nn.Conv2d(8, 3, 3, padding=1),  
            nn.Sigmoid()
        )

    def forward(self, x):
        # print(f"Received input Data with shape: {x.shape}")
        x = self.encoder(x)
        # print(f"Encoder finished and has Latent Space Representation of shape: {x.shape}")
        x = self.decoder(x)
        # print(f"Decoder finished and output has shape: {x.shape}")
        return x

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

class CAE_2(nn.Module):
    """
    Decrease Latent Space
    """
    def __init__(self):
        super(CAE_2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, padding=0),
            nn.Conv2d(8, 12, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, padding=0),
            nn.Conv2d(12, 4, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(4, 12, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(12, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print(f"Received input Data with shape: {x.shape}")
        x = self.encoder(x)
        # print(f"Encoder finished and has Latent Space Representation of shape: {x.shape}")
        x = self.decoder(x)
        # print(f"Decoder finished and output has shape: {x.shape}")
        return x

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

'''Convolutional Autoencoder for grayscale-to-color conversion using same architecture we found in 2'''
class CAE_1_Color(nn.Module):
    """
    Same Latent space but more informed Dimensionality Reduction and Increase and modified for single channel
    """
    def __init__(self):
        super(CAE_1_Color, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 12, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(12, 16, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 12, 3, padding=1,stride=2, output_padding=1),  # Doubles spatial dimensions
            nn.ReLU(),
            nn.ConvTranspose2d(12, 8, 3, padding=1,stride=2, output_padding=1),  
            nn.ReLU(),
            nn.Conv2d(8, 3, 3, padding=1),  
            nn.Sigmoid()
        )

    def forward(self, x):
        # print(f"Received input Data with shape: {x.shape}")
        x = self.encoder(x)
        # print(f"Encoder finished and has Latent Space Representation of shape: {x.shape}")
        x = self.decoder(x)
        # print(f"Decoder finished and output has shape: {x.shape}")
        return x

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

'''Convolutional Autoencoder for predicting chrominance components from luminance using mainly the same architecture we found in 2'''
class CAE_Chrominance(nn.Module):
    def __init__(self):
        super(CAE_Chrominance, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, stride=2),  # Grayscale input (1 channel)
            nn.ReLU(),
            nn.Conv2d(8, 12, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(12, 16, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 12, 3, padding=1, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 8, 3, padding=1, stride=2, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 2, 3, padding=1),  
            nn.Tanh()  
        )


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print('Model saved at {}'.format(path))


def load_model(path, model_class):
    loaded_model = model_class()
    loaded_model.load_state_dict(torch.load(path))
    loaded_model.eval()  
    print("Model loaded and ready for inference.")
    return loaded_model
    

def imshow(img):
    img = img  
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

'''Creates and returns data loaders for training and testing given each respective EncoderType'''
def get_DataLoaders(EncoderType):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='', train=True, download=False, transform=transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root='', train=False, download=False, transform=transform
    )

    full_dataset = ConcatDataset([train_dataset, test_dataset])

    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size  # Ensure all samples are used

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    train_autoencoder_dataset = EncoderType(train_dataset)
    val_autoencoder_dataset = EncoderType(val_dataset)
    test_autoencoder_dataset = EncoderType(test_dataset)

    batch_size = 128

    train_loader = DataLoader(
        train_autoencoder_dataset,
        batch_size=batch_size,
        shuffle=True, 
        num_workers=2  # number of subprocesses for loading data
    )

    test_loader = DataLoader(
        test_autoencoder_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    return train_loader, test_loader


'''Trains the given model_type on train set and saves the trained model+history'''
def train_model(model_type,train_loader, model_path="Model.pth",loss_history_path="loss_history.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    print(f"Using device: {device}")

    model = model_type().to(device)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    epochs = 20
    loss_history = []
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        epoch_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            epoch_loss += loss.item()
            if i % 100 == 99:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        loss_history.append(epoch_loss / len(train_loader))
    print('Finished Training')
    save_model(model, model_path)
    print("Plot Loss History during Training")

    import matplotlib.pyplot as plt

    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss History')
    plt.savefig(loss_history_path)
    plt.show()
    return model, loss_history


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

'''Reconstructs a color image from grayscale and predicted chrominance components'''
def reconstruct_color_image(grayscale, predicted_uv):
    # Convert grayscale and predicted_uv to numpy arrays (H, W)
    y_channel = grayscale.detach().numpy()[0] * 255  
    u_channel = predicted_uv.detach().numpy()[0] * 127.5 + 128  # [-1, 1] -> [0, 255]
    v_channel = predicted_uv.detach().numpy()[1] * 127.5 + 128  # [-1, 1] -> [0, 255]

    # Stack Y, U, V channels into the YCbCr color space
    yuv_image = np.stack((y_channel, u_channel, v_channel), axis=-1)

    # Convert YCbCr to RGB
    yuv_image = yuv_image.astype(np.uint8)
    yuv_image_pil = Image.fromarray(yuv_image, 'YCbCr')
    rgb_image_pil = yuv_image_pil.convert('RGB')
    rgb_image = np.array(rgb_image_pil)

    plt.imshow(rgb_image)
    plt.axis('off')  
    plt.show()

    return rgb_image


'''Computes and returns the average test error over the test set'''
def compute_test_error(model, test_loader, criterion):
    model.eval()  
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():  
        for inputs, labels in tqdm(test_loader):

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)  
            total_samples += inputs.size(0)  

   
    avg_loss = total_loss / total_samples
    return avg_loss

if __name__ == '__main__':
    train_loader, test_loader = get_DataLoaders(ChrominanceEncoderDataset)

    model, history_loss = train_model(CAE_Chrominance, train_loader, model_path="cae_chrominance.pth",
                                      loss_history_path="cae_chrominance_loss.png")

    model = model.cpu()
    model.eval()

    grayscale, _ = test_loader.dataset[0]  
    grayscale = grayscale.unsqueeze(0)  

    # Predict UV channels
    with torch.no_grad():
        predicted_uv = model(grayscale)

    grayscale = grayscale.squeeze(0)
    predicted_uv = predicted_uv.squeeze(0)

    color_image = reconstruct_color_image(grayscale, predicted_uv)

    criterion = nn.MSELoss()
    test_error = compute_test_error(model, test_loader, criterion)
    print(f'Test Error (MSE Loss): {test_error:.4f}')
