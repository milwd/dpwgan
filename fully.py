
import pandas as pd
import logging
import torch
import torch.utils.data
# import torch.nn.functional as F
from backend import *
_logger = logging.getLogger(__name__)


### preprocess functions of datasets ###

def preprocess_adult(addr):
    df = pd.read_csv(addr, 
                index_col=False,
                names=["age", 
                    "workclass",
                    "fnlwgt", 
                    "education", 
                    "education-num", 
                    "marital-status", 
                    "occupation", 
                    "relationship", 
                    "race", 
                    "sex", 
                    "capital-gain", 
                    "capital-loss", 
                    "hours-per-week", 
                    "native-country", 
                    "income"])
    df = df[["capital-gain", 
            "capital-loss", 
            "hours-per-week",
            "age",

            "workclass",
            "education", 
            "marital-status", 
            "occupation", 
            "relationship", 
            "race", 
            "sex", 
            "native-country", 
            "income"]]
    continuous_columns = ["capital-gain", "capital-loss", "hours-per-week", "age"]
    return df, continuous_columns


def preprocess_obesity(addr):
    obesity = pd.read_csv(addr)
    print(obesity.columns)
    df = obesity[['Age', 
                    'Height', 
                    'Weight', 
                    'FCVC',
                    'NCP',
                    'CH2O', 
                    'FAF', 
                    'TUE', 

                    'Gender', 
                    'family_history_with_overweight', 
                    'FAVC', 
                    'CAEC', 
                    'SMOKE',                     
                    'SCC', 
                    'CALC', 
                    'MTRANS', 
                    'NObeyesdad']]
    continuous_columns = ['Age', 
                        'Height', 
                        'Weight', 
                        'FCVC',
                        'NCP',
                        'CH2O', 
                        'FAF', 
                        'TUE']
    return df, continuous_columns


def preprocess_mushroom(addr):
    mushroom = pd.read_csv(addr, 
                        index_col=False,
                        names=["poison", 
                                "cap-shape", 
                                "cap-surface", 
                                "cap-color", 
                                "bruises", 
                                "odor", 
                                "gill-attachment", 
                                "gill-spacing", 
                                "gill-size", 
                                "gill-color", 
                                "stalk-shape", 
                                "stalk-root", 
                                "stalk-surface-above-ring", 
                                "stalk-surface-below-ring", 
                                "stalk-color-above-ring", 
                                "stalk-color-below-ring", 
                                "veil-type", 
                                "veil-color", 
                                "ring-number", 
                                "ring-type", 
                                "spore-print-color", 
                                "population", 
                                "habitat"])
    df = mushroom[["cap-shape", 
                        "cap-surface", 
                        "cap-color", 
                        "bruises", 
                        "odor", 
                        "gill-attachment", 
                        "gill-spacing", 
                        "gill-size", 
                        "gill-color", 
                        "stalk-shape", 
                        "stalk-root", 
                        "stalk-surface-above-ring", 
                        "stalk-surface-below-ring", 
                        "stalk-color-above-ring", 
                        "stalk-color-below-ring", 
                        "veil-type", 
                        "veil-color", 
                        "ring-number", 
                        "ring-type", 
                        "spore-print-color", 
                        "population", 
                        "habitat", 
                        "poison"]]
    df.drop(df[df['stalk-root'] == '?'].index, inplace = True)  # 2480 rows
    return df, []  # no continuous columns


### generate a synthetic categorical dataset ###

def generate_data():
    df = pd.DataFrame(
        {'weather': ['sunny']*10000+['cloudy']*10000+['rainy']*10000,
         'status': 
           ['on time']*8000+['delayed']*2000
         + ['on time']*3000+['delayed']*5000+['canceled']*2000
         + ['on time']*2000+['delayed']*4000+['canceled']*4000}
    )
    return df


def main():
    torch.manual_seed(123)
    logging.basicConfig(level=logging.INFO)

    NOISE_DIM = 10
    HIDDEN_DIM = 20
    SIGMA = 0.5

    datasetnames        = ["adult", "obesity", "mushroom"]
    preprocess_datasets = [preprocess_adult, preprocess_obesity, preprocess_mushroom]

    real_data = Dataset(datasetnames, preprocess_datasets, "mushroom", "datasets/agaricus-lepiota.data")
    data_tensor = real_data.df_torch

    gan = create_categorical_gan(NOISE_DIM, 
                                HIDDEN_DIM, 
                                [[len(real_data.continuous_columns)], real_data.categorical_dimensions]) 
    
    gan.train(data=data_tensor,
              epochs=100,
              n_critics=5,
              batch_size=64,
              learning_rate=3e-4,
              weight_clip=1/HIDDEN_DIM,
              sigma=SIGMA)
    
    flat_synth_data = gan.generate(200)
    output = real_data.scaleup(flat_synth_data)

    print(output.iloc[0])
    output.to_csv("mushroom_sigma1_epoch20.csv", sep=',', index=False)

    # print('Real data crosstab:')
    # print(percentage_crosstab(real_data['weather'], real_data['status']))
    # print('Synthetic data crosstab:')
    # print(percentage_crosstab(synth_data['weather'], synth_data['status']))


if __name__ == '__main__':
    main()
