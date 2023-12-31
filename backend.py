import collections
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import pandas as pd
import torch
import torch.utils.data



class MultiCategoryGumbelSoftmax(torch.nn.Module):
    """Gumbel softmax for multiple output categories

    Parameters
    ----------
    input_dim : int
        Dimension for input layer
    output_dims : list of int
        Dimensions of categorical output variables
    tau : float
        Temperature for Gumbel softmax
    """
    def __init__(self, input_dim, output_dims, tau=2/3):
        super(MultiCategoryGumbelSoftmax, self).__init__()
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(input_dim, output_dim)
            for output_dim in output_dims
        )
        self.tau = tau

    def gumbel_softmax(self, logits, tau=2/3, hard=False, dim=-1):
        def _gen_gumbels():
            gumbels = -torch.empty_like(logits).exponential_().log()
            if torch.isnan(gumbels).sum() or torch.isinf(gumbels).sum():
                gumbels = _gen_gumbels()
            return gumbels
        gumbels = _gen_gumbels()  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim)
        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret

    def forward(self, x):
        xs = tuple(layer(x) for layer in self.layers)
        logits = tuple(nn.functional.log_softmax(x, dim=1) for x in xs)
        categorical_outputs = tuple(
            self.gumbel_softmax(logit, tau=self.tau, hard=True)
            for logit in logits
        )
        return torch.cat(categorical_outputs, 1)


class DPWGAN(object):
    """Class to store, train, and generate from a
    differentially-private Wasserstein GAN

    Parameters
    ----------
    generator : torch.nn.Module
        torch Module mapping from random input to synthetic data

    discriminator : torch.nn.Module
        torch Module mapping from data to a real value

    noise_function : function
        Mapping from number of samples to a tensor with n samples of random
        data for input to the generator. The dimensions of the output noise
        must match the input dimensions of the generator.
    """
    def __init__(self, generator, discriminator, noise_function):
        self.generator = generator
        self.discriminator = discriminator
        self.noise_function = noise_function

    def train(self, data, epochs=100, n_critics=5, batch_size=128,
              learning_rate=1e-4, sigma=None, weight_clip=0.1):
        """Train the model

        Parameters
        ----------
        data : torch.Tensor
            Data for training
        epochs : int
            Number of iterations over the full data set for training
        n_critics : int
            Number of discriminator training iterations
        batch_size : int
            Number of training examples per inner iteration
        learning_rate : float
            Learning rate for training
        sigma : float or None
            Amount of noise to add (for differential privacy)
        weight_clip : float
            Maximum range of weights (for differential privacy)
        """
        generator_solver = optim.RMSprop(
            self.generator.parameters(), lr=learning_rate
        )
        discriminator_solver = optim.RMSprop(
            self.discriminator.parameters(), lr=learning_rate
        )

        # add hooks to introduce noise to gradient for differential privacy
        if sigma is not None:
            for parameter in self.discriminator.parameters():
                parameter.register_hook(
                    lambda grad: grad + (1 / batch_size) * sigma
                    * torch.randn(parameter.shape)
                )

        # There is a batch for each critic (discriminator training iteration),
        # so each epoch is epoch_length iterations, and the total number of
        # iterations is the number of epochs times the length of each epoch.
        epoch_length = len(data) / (n_critics * batch_size)
        n_iters = int(epochs * epoch_length)
        for iteration in range(n_iters):
            for _ in range(n_critics):
                # Sample real data
                rand_perm = torch.randperm(data.size(0))
                samples = data[rand_perm[:batch_size]]
                real_sample = Variable(samples)

                # Sample fake data
                fake_sample = self.generate(batch_size)

                # Score data
                discriminator_real = self.discriminator(real_sample)
                discriminator_fake = self.discriminator(fake_sample)

                # Calculate discriminator loss
                # Discriminator wants to assign a high score to real data
                # and a low score to fake data
                discriminator_loss = -(
                    torch.mean(discriminator_real) -
                    torch.mean(discriminator_fake)
                )

                discriminator_loss.backward()
                discriminator_solver.step()

                # Weight clipping for privacy guarantee
                for param in self.discriminator.parameters():
                    param.data.clamp_(-weight_clip, weight_clip)

                # Reset gradient
                self.generator.zero_grad()
                self.discriminator.zero_grad()

            # Sample and score fake data
            fake_sample = self.generate(batch_size)
            discriminator_fake = self.discriminator(fake_sample)

            # Calculate generator loss
            # Generator wants discriminator to assign a high score to fake data
            generator_loss = -torch.mean(discriminator_fake)

            generator_loss.backward()
            generator_solver.step()

            # Reset gradient
            self.generator.zero_grad()
            self.discriminator.zero_grad()

            # Print training losses
            if int(iteration % epoch_length) == 0:
                epoch = int(iteration / epoch_length)
                print('Epoch ' + str(epoch) + \
                    ' | Disc loss: ' + str(np.round(discriminator_loss.data.numpy(), 5)) + \
                    ' | Genr loss: ' + str(np.round(generator_loss.data.numpy(), 5))
                )

    def generate(self, n):
        """Generate a synthetic data set using the trained model

        Parameters
        ----------
        n : int
            Number of data points to generate

        Returns
        -------
        torch.Tensor
        """
        noise = self.noise_function(n)
        fake_sample = self.generator(noise)
        return fake_sample


class Generator(nn.Module):
    def __init__(self, noise_dim, hidden_dim, output_dims):
        super().__init__()
        self.fc1 = nn.Linear(noise_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dims[0][0])
        self.mcg = MultiCategoryGumbelSoftmax(hidden_dim, output_dims[1])
        self.rlu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.rlu(x)
        x_categorical = self.mcg(x)
        x_continuous  = self.fc2(x)
        return torch.cat([x_continuous, x_categorical], 1)


class Dataset:
    def __init__(self, datasetnames, preprocess_datasets, nameofdataset, addr):
        
        self.datasetnames, self.preprocess_datasets = datasetnames, preprocess_datasets

        self.read(addr, nameofdataset)
        self.categorical_columns = [i for i in self.df.columns if i not in self.continuous_columns]
        self.categorical_dataset(self.df[self.categorical_columns]) 
        categorical_torch = self.to_onehot_flat(self.df[self.categorical_columns])

        # TODO add if only continuous
        if len(self.continuous_columns) > 1:  # if categorical + continuous
            self.ss = StandardScaler()
            self.scaledown()
            continuous_torch  = torch.Tensor(self.df[self.continuous_columns].values)
            self.df_torch = torch.cat([continuous_torch, categorical_torch], 1)
        else:                                 # if only categorical
            self.df_torch = categorical_torch 

    def read(self, addr, nameofdataset):
        self.df, self.continuous_columns = self.preprocess_datasets[self.datasetnames.index(nameofdataset)](addr)
        
    def scaledown(self):
        cont_scaled = self.ss.fit_transform(self.df[self.continuous_columns].to_numpy().reshape(-1, len(self.continuous_columns)))
        self.df[self.continuous_columns] = pd.DataFrame(cont_scaled, columns=self.continuous_columns)

    def categorical_dataset(self, data):
        self.codes = collections.OrderedDict((var, np.unique(data[var])) for var in data)
        self.categorical_dimensions = [len(code) for code in self.codes.values()]

    def to_onehot_flat(self, data):
        """
        Returns a torch Tensor with onehot encodings of each variable
        in the categorical dataset

        Returns
        -------
        torch.Tensor
        """
        return torch.cat([to_onehot(data[var], code) for var, code in self.codes.items()], 1)

    def from_onehot_flat(self, data):
        """
        Converts from a torch Tensor with onehot encodings of each variable
        to a pandas DataFrame with categories

        Parameters
        ----------
        data : torch.Tensor

        Returns
        -------
        pandas.DataFrame
        """
        categorical_data = pd.DataFrame()
        index = len(self.continuous_columns)
        for var, code in self.codes.items():
            var_data = data[:, index:(index+len(code))]
            categorical_data[var] = from_onehot(var_data, code)
            index += len(code)
        return categorical_data

    def scaleup(self, data):
        synth_data_categorical = self.from_onehot_flat(data) 
        if len(self.continuous_columns) > 1:
            synth_data_continuous  = pd.DataFrame(self.ss.inverse_transform(data[:, :len(self.continuous_columns)].detach().numpy()), columns=self.continuous_columns)
            return pd.concat([synth_data_continuous, synth_data_categorical], axis=1)
        else:
            return synth_data_categorical


def create_categorical_gan(noise_dim, hidden_dim, output_dims):
    generator = Generator(noise_dim, hidden_dim, output_dims)
    discriminator = torch.nn.Sequential(
        torch.nn.Linear(output_dims[0][0]+sum(output_dims[1]), hidden_dim),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(hidden_dim, 1)
    )

    def noise_function(n):
        return torch.randn(n, noise_dim)

    gan = DPWGAN(
        generator=generator,
        discriminator=discriminator,
        noise_function=noise_function
    )

    return gan


def percentage_crosstab(variable_one, variable_two):
    return 100*pd.crosstab(variable_one, variable_two).apply(
        lambda r: r/r.sum(), axis=1
    )


def to_onehot(data, codes):
    """
    data: column of categorical variable
    codes: unique values of data column
    """
    indices = [np.where(codes == val)[0][0] for val in data]
    indices = torch.LongTensor(list([val] for val in indices))
    onehot = torch.FloatTensor(indices.size(0), len(codes)).zero_()
    onehot.scatter_(1, indices, 1)
    return onehot


def from_onehot(data, codes):
    return codes[[np.where(data[i] == 1)[0][0] for i in range(len(data))]]
            