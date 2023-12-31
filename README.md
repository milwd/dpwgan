# Differentially-Private WGANs


The original code runs great. However since the original code is released as-is, and is not actively maintained, there were several issues that has been resolved here:

* The discriminator's loss sometimes underflows which results in 'nan'. Subsequently, the generator's loss becomes 'nan' and the model stops learning. I've used a slightly altered GumbleSoftmax from [a here](https://gist.github.com/GongXinyuu/3536da55639bd9bfdd5a905ebf3ab88e) which resolves the issue. 

* Initially, the code was specific to only-categorical datasets. I wanted this code for datasets with categorical and continuous features like [a Adult](https://archive.ics.uci.edu/dataset/20/census+income) and [a Obesity](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition). The added ```Dataset``` class incorporates mixing categorical and continuous datasets. The new Generator is bi-headed (Softmax for categorical and ReLU for continuous). 

* I have also restructured the code, which is more readable for me. 


## Installation

This package requires Python >= 3.5

```
pip install -r requirements.txt
python setup.py install
```

For development, also install development requirements:

```
pip install -r dev-requirements.txt
```


## Usage

The main run code is in ```fully.py``` and the classes and functions are stored in ```backend.py```.

Add a function ```def preprocess_adult(addr):``` to preprocess your dataset and return a ```DataFrame```, and a ```list of the names of continuous features```. 
Remember to put all the continuous features before categorical features. 
Include a name for your dataset in ```datasetnames``` and include your preprocess function name in ```preprocess_datasets```. Use the name and the address of the csv file for Dataset obejct instantiation.

```
datasetnames        = ["adult", "obesity", "mushroom"]
preprocess_datasets = [preprocess_adult, preprocess_obesity, preprocess_mushroom]

real_data = Dataset(datasetnames, preprocess_datasets, "mushroom", "datasets/agaricus-lepiota.data")
```

Setting up, training, and generating from a DPWGAN:

```
gan = DPWGAN(generator, discriminator, noise_function)
gan.train(data)
synthetic_data = gan.generate(100)
```

`generator` and `discriminator` should be `torch.nn` modules, and
`noise_function` should generate random data as input to the `generator`.
As a simple example:

```
# simple generator module
generator = torch.nn.Sequential(
    torch.nn.Linear(noise_dim, hidden_dim),
    torch.nn.ReLU(),
    MultiCategoryGumbelSoftmax(hidden_dim, output_dims)
)

# simple discriminator module
discriminator = torch.nn.Sequential(
    torch.nn.Linear(sum(output_dims), hidden_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_dim, 1)
)

# simple noise function (input to generator module)
def noise_function(n):
    return torch.randn(n, noise_dim)
```
