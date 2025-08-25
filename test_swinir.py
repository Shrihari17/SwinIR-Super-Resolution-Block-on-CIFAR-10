import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from swinir import SwinIRBlock
import os

def get_cifar10_loader(batch_size=8, data_dir="./cifar-10-batches-py"):
    transform = T.Compose([
        T.Resize((32,32)),
        T.ToTensor(),
    ])
    download = not os.path.exists(data_dir)
    dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=download, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def visualize(inputs, outputs, n=6):
    plt.figure(figsize=(12,4))
    for i in range(n):
        plt.subplot(2,n,i+1)
        plt.imshow(inputs[i].permute(1,2,0))
        plt.axis("off")
        plt.title("Input")
        plt.subplot(2,n,n+i+1)
        plt.imshow(outputs[i].permute(1,2,0).detach())
        plt.axis("off")
        plt.title("Output")
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CIFAR-10
    loader = get_cifar10_loader(batch_size=8)
    imgs, _ = next(iter(loader))
    imgs = imgs.to(device)

    # Create SwinIR block
    model = SwinIRBlock(dim=96, input_resolution=(32,32), num_heads=3, window_size=8, shift_size=4).to(device)

    # Forward pass
    outs = model(imgs)

    # Visualize
    visualize(imgs.cpu(), outs.cpu(), n=6)
    print("Input shape:", imgs.shape)
    print("Output shape:", outs.shape)

if __name__ == "__main__":
    main()
