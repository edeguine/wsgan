import torch
import torch.nn as nn
from torch import optim

from skimage import io
import numpy

import LoadData

fake_label = 0
real_label = 1
cuda = True 
datasetVersion = 'dataset'

class Config():
    def __init__(self, batchSize):
        self.epochs = 50000
        self.batchSize = batchSize
        self.nc = 3
        self.zx = 9
        self.zxSample = 20
        self.nz = 100

        self.l2Fac = 1e-5 # L2 reg

        self.genLayers = 5
        self.disLayers = self.genLayers

        self.genKernelSize = [(5, 5)] * self.genLayers
        self.genKernelSize = self.genKernelSize[::-1]

        self.disKernelSize = self.genKernelSize

        self.genOutChannels = [self.nc] + [2 ** (n + 6) for n in range(self.genLayers - 1)]
        self.genOutChannels = self.genOutChannels[::-1]

        self.genInChannels = [self.nz] + self.genOutChannels[:-1]
        self.genPadding = 2


        self.disOutChannels = [2 ** (n + 6) for n in range(self.disLayers - 1)] + [1] # TODO check 1 or 2 output channels for final layer

        self.disInChannels = [self.nc] + self.disOutChannels[:-1]
        self.disPadding = 2

        self.genStride = [(2, 2)] * self.genLayers # Dimensions: NC * W * H input, then genOutChannels[i] * (W * 2 ^ (i + 1)) * (H * 2 ^ (i + 1)), last is npx = zx * 32
        self.disStride = [(2, 2)] * self.disLayers # Dimensions last layer is Wgen / 2**5 = Wgen / 32 = npx

        self.lr = 0.0005
        self.b1 = 0.5
        self.l2_fac = 1e-5

        self.Dupdates = 1
        self.npx = zxToNpx(self.zx, self.genLayers)

def zxToNpx(zx, depth):
    return (zx - 1) * 2 ** depth + 1

class NetG(nn.Module):
    def __init__(self, config):
        super(NetG, self).__init__()

        self.layers = torch.nn.ModuleList()

        # Transposed Convolution
        # outChannels - num_filters - gen_fn
        # kernelSize - filter_size - gen_ks
        # stride - (2, 2)

        # Batchnorm

        print(config.zx)


        for l in range(config.genLayers - 1):
            tconv = torch.nn.ConvTranspose2d(
                config.genInChannels[l],
                config.genOutChannels[l],
                config.genKernelSize[l],
                stride = config.genStride[l],
                padding=config.genPadding)

            activation = torch.nn.ReLU()

            bnorm = torch.nn.BatchNorm2d(
                config.genOutChannels[l])

            self.layers.append(tconv)
            self.layers.append(bnorm)
            self.layers.append(activation)

        tconv = torch.nn.ConvTranspose2d(
            config.genInChannels[-1],
            config.genOutChannels[-1],
            config.genKernelSize[-1],
            stride = config.genStride[-1],
            padding=config.genPadding)

        activation = torch.nn.Tanh()

        self.layers.append(tconv)
        self.layers.append(activation)

    def forward(self, x):
        out = self.layers[0](x)
        for i, l in enumerate(self.layers[1:]):
            out = l(out)
        return out

class NetD(nn.Module):
    def __init__(self, config):
        super(NetD, self).__init__()

        self.layers = torch.nn.ModuleList()
        first = torch.nn.Conv2d(
            config.disInChannels[0],
            config.disOutChannels[0],
            config.disKernelSize[0],
            stride = config.disStride[0],
            padding = config.disPadding)
        
        firstActivation = torch.nn.LeakyReLU(negative_slope=0.2)
        self.layers.append(first)
        self.layers.append(firstActivation)

        for l in range(1, config.disLayers - 1):
            conv = torch.nn.Conv2d(
                config.disInChannels[l],
                config.disOutChannels[l],
                config.disKernelSize[l],
                stride = config.disStride[l],
                padding = config.disPadding)

            activation = torch.nn.LeakyReLU(negative_slope = 0.2)
            bnorm = torch.nn.BatchNorm2d(config.disOutChannels[l])
            
            self.layers.append(conv)
            self.layers.append(bnorm)
            self.layers.append(activation)

        last = torch.nn.Conv2d(
            config.disInChannels[-1],
            config.disOutChannels[-1],
            config.disKernelSize[-1],
            stride = config.disStride[-1],
            padding = config.disPadding)

        lastActivation = torch.nn.Sigmoid()

        self.layers.append(last)
        self.layers.append(lastActivation)

    def forward(self, x):
        out = self.layers[0](x)
        for l in self.layers[1:]:
            out = l(out)
        return out


def lossD(predReal, predFake):
    shape = predReal.shape
    temp = predFake[:predReal.shape[0], :, :, :] # This is to accomodate partial batches
    ones = torch.ones(shape)
    if cuda:
        ones = ones.cuda()
    res = -torch.mean(torch.log(ones - temp)) - torch.mean(torch.log(predReal))
    return res

def lossG(predFake):
    shape = predFake.shape
    res = -torch.mean(torch.log(predFake))
    return res 

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def main():
    
    config = Config(6)


    if cuda:
        torch.backends.cudnn.deterministic=True

    gen = NetG(config)
    if cuda:
        gen = gen.cuda()
    dis = NetD(config)
    if cuda:
        dis = dis.cuda()

    weights_init(gen)
    weights_init(dis)

    print(gen)
    print(dis)

    trainset = LoadData.loadDataset(datasetVersion, config.npx)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batchSize, shuffle=True, num_workers=2)

    optimizerG = optim.Adam(gen.parameters(), lr=config.lr, betas=(config.b1, 0.999))#, weight_decay=config.l2Fac)
    optimizerD = optim.Adam(dis.parameters(), lr=config.lr, betas=(config.b1, 0.999))#, weight_decay=config.l2Fac)

    # Fixed noise used to generate the samples, slightly larger than the one used for training
    # A larger sample helps get a sense of the final large scale results

    Zsample = torch.FloatTensor(1, config.nz, config.zxSample, config.zxSample).uniform_(-1, 1)
    if cuda:
        Zsample = Zsample.cuda()

    for epoch in range(config.epochs):

        errG = []
        errD = []

        it = 0
        for i, data in enumerate(trainloader):
            
            if it % (config.Dupdates + 1) == 0:

                # Training the generator
                gen.zero_grad()

                # Generate the noise
                Z = torch.FloatTensor(config.batchSize, config.nz, config.zx, config.zx).uniform_(-1, 1)
                if cuda:
                    Z = Z.cuda()

                genX = gen(Z)
                dFake = dis(genX)

                # Compute the loss on fake
                og = lossG(dFake)
                og.backward()

                optimizerG.step()
                errG.append(og.item())
            else:
                # Training the discriminator
                dis.zero_grad()

                inputs, labels = data
                inputs = inputs.float()
                if cuda:
                    inputs = inputs.cuda()

                Z = torch.FloatTensor(config.batchSize, config.nz, config.zx, config.zx).uniform_(-1, 1)
                if cuda:
                    Z = Z.cuda()

                X = inputs
                genX = gen(Z)

                # Compute real and fake labels
                dFake = dis(genX)
                dReal = dis(X)

                # Compute the loss
                od = lossD(dReal, dFake)
                od.backward()
                optimizerD.step()
                errD.append(od.item())

            it += 1

        print(f"Epoch {epoch} Losses: G = {numpy.mean(errG)} D = {numpy.mean(errD)}")
        
        # Save a smallish sample
        if epoch % 5 == 0:
            with torch.no_grad():
                fake = gen(Zsample).detach().cpu().numpy()[0, :, :, :]
                im = numpy.zeros((fake.shape[1], fake.shape[2], fake.shape[0]))
                im[:, :, 0] = fake[0, :, :]
                im[:, :, 1] = fake[1, :, :]
                im[:, :, 2] = fake[2, :, :]
                io.imsave(f"samples/torch_sample_{datasetVersion}_{epoch}.png", im)

        # Save the model
        if epoch % 50 == 0:
            torch.save(gen.state_dict(), f"models/gen_{epoch}.pth")
            torch.save(dis.state_dict(), f"models/dis_{epoch}.pth")

if __name__ == "__main__":
    main()


