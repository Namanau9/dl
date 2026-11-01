import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

z_dim=64
batch_size=64
device="cuda" if torch.cuda.is_available() else "cpu"

transform=transforms.Compose([
transforms.ToTensor(),
transforms.Normalize([0.5],[0.5])])

loader=torch.utils.data.DataLoader(
datasets.MNIST("./data",train=True,download=True,transform=transform),
batch_size=batch_size,shuffle=True)

class Generator(nn.Module):
    def __init__(self,z_dim=64,img_dim=28*28):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(z_dim,128),
            nn.ReLU(),
            nn.Linear(128,img_dim),
            nn.Tanh())
    def forward(self,z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self,img_dim=28*28):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(img_dim,128),
            nn.LeakyReLU(0.2),
            nn.Linear(128,1))
    def forward(self,x):
        return self.net(x)

G=Generator(z_dim).to(device)
D=Discriminator().to(device)

criterion=nn.MSELoss()
opt_G=optim.Adam(G.parameters(),lr=0.0002)
opt_D=optim.Adam(D.parameters(),lr=0.0002)

for epoch in range(1):
    for i,(real,_) in enumerate(loader):
        real=real.view(-1,28*28).to(device)
        b=real.size(0)

        z=torch.randn(b,z_dim).to(device)
        fake=G(z).detach()
        D_real=D(real)
        D_fake=D(fake)

        loss_D=.5*(criterion(D_real,torch.ones_like(D_real))+criterion(D_fake,torch.zeros_like(D_fake)))

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        z=torch.randn(b,z_dim).to(device)
        fake=G(z)
        D_fake=D(fake)

        loss_G=.5*criterion(D_fake,torch.ones_like(D_fake))

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        if i%200==0:
            print(f"Step {i}: D_loss={loss_D.item():.4f}, G_loss={loss_G.item():.4f}")
        if i>=200:
            break

    with torch.no_grad():
        samples=G(torch.randn(16,z_dim).to(device)).cpu()

    grid=make_grid(samples.view(-1,1,28,28),nrow=4,normalize=True)
    plt.imshow(grid.permute(1,2,0).numpy())
    plt.axis("off")
    plt.show()
