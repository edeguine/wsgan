import sganTorch as sgan
import sys
import torch
import numpy
from skimage import io

cuda = False

# Code to generate large textures based on the generator model

def main():
    
    config = sgan.Config(6)
    gen = sgan.NetG(config)
    if cuda:
        gen = gen.cuda()

    if cuda:
        gen.load_state_dict(torch.load(sys.argv[1]))
    else:
        gen.load_state_dict(torch.load(sys.argv[1], map_location=torch.device('cpu')))

    genZX = 60

    modelName = sys.argv[1].split('/')[-1].split('.')[0]

    for i in range(5):
        noise = torch.FloatTensor(1, config.nz, genZX, genZX).uniform_(-1, 1)
        if cuda:
            noise = noise.cuda()
        output = gen(noise)
        imfake = output.detach().cpu().numpy()[0, :, :, :]
        im = numpy.zeros((imfake.shape[1], imfake.shape[2], imfake.shape[0]))
        im[:, :, 0] = imfake[0, :, :]
        im[:, :, 1] = imfake[1, :, :]
        im[:, :, 2] = imfake[2, :, :]
        io.imsave(f"samples/stored_{sgan.datasetVersion}_{modelName}_{i}.png", im)

if __name__ == "__main__":
    main()
