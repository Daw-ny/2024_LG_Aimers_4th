from module.aae import *
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import torch
import warnings
import itertools
import pandas as pd

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    train = pd.read_csv('../Database/test/train_k.csv', index_col=0)
    X = train.drop(columns=['is_converted'])
    y = train['is_converted']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # ----------------------------------------------------------------------------------------------------------------#
    X_train_tensor = torch.tensor(X_train.values)
    y_train_tensor = torch.tensor(y_train.values)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    encoder = Encoder(32)
    decoder = Decoder(32)
    discriminator = Discriminator()
    adversarial_loss = torch.nn.BCELoss()
    reconstruction_loss = torch.nn.L1Loss()

    cuda = True if torch.cuda.is_available() else False
    if cuda:
        encoder.cuda()
        decoder.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        reconstruction_loss.cuda()

    optimizer_G = torch.optim.Adam(
        itertools.chain(encoder.parameters(), decoder.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
    )
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    for epoch in range(opt.n_epochs):
        for data in dataloader:
            real_data = data[0].float().cuda()
            # real_data = Variable(data)
            batch_size = real_data.size(0)
            real_labels = torch.ones(batch_size, 1).cuda()
            # real_labels = Variable(torch.ones(batch_size, 1))
            fake_labels = torch.zeros(batch_size, 1).cuda()
            # fake_labels = Variable(torch.zeros(batch_size, 1))

            z = encoder(real_data)
            reconstructed_data = decoder(z)
            g_loss = (0.001 * adversarial_loss(discriminator(z), real_labels) +
                      0.999 * reconstruction_loss(reconstructed_data, real_data))

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            real_loss = adversarial_loss(discriminator(z), real_labels)
            fake_loss = adversarial_loss(discriminator(reconstructed_data.detach()), fake_labels)
            discriminator_loss = (real_loss + fake_loss) * 0.5

            optimizer_D.zero_grad()
            discriminator_loss.backward()
            optimizer_D.step()

# with torch.no_grad():
#     new_z = torch.randn(64, 1)
# generated_data = decoder(new_z)
