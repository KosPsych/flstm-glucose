from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.impute import KNNImputer
from dataset import *
from torch.utils.data import random_split
import joblib





def outlier_removal(df):
    df['gsr'] = df['gsr'].apply(lambda x: min(x, 25))
    df['carbInput'] = df['carbInput'].apply(lambda x: min(x, 250))
    return df


def first_imputation(df):

    # Imputation
    df['carbInput'] = df['carbInput'].fillna(method='ffill').fillna(0)
    df['bolus'] = df['bolus'].fillna(method='ffill').fillna(0)
    df['basal'] = df['basal'].fillna(method='ffill').fillna(0)

    df.loc[df['missing_cbg'] == 1, 'cbg'] = df['finger']
    df.drop(columns=['finger', 'hr', 'missing_cbg', '5minute_intervals_timestamp'], inplace = True)
    return df


def second_imputation_and_scaling(df_patient1, df_patient2, imputer, scaler):

    df = pd.concat([df_patient1, df_patient2])

    scaled_df = scaler.fit_transform(df)

    imputed_df = pd.DataFrame(imputer.fit_transform(scaled_df), columns=df_patient1  .columns)





    return imputed_df[imputed_df['id'] == 0], imputed_df[imputed_df['id'] == 1]



def preprocess_test(df_patient1, df_patient2):
    # Sorting based on time feature
    df_patient1 = df_patient1.sort_values(by='5minute_intervals_timestamp')
    df_patient2 = df_patient2.sort_values(by='5minute_intervals_timestamp')

    # first imputation
    df_patient1 = first_imputation(df_patient1)
    df_patient2 = first_imputation(df_patient2)

    from constants import SAVE_PATH
    imputer = joblib.load(SAVE_PATH + 'imputer_model.joblib')
    scaler = joblib.load(SAVE_PATH + 'scaler_model.joblib')



    df_patient1['id'] = 0
    df_patient2['id'] = 0

    columns = df_patient1.columns
    df_patient1 = pd.DataFrame(scaler.transform(df_patient1), columns = columns)
    df_patient2 = pd.DataFrame(scaler.transform(df_patient2), columns = columns)


    cbg_pat1 = df_patient1['cbg']
    cbg_pat2 = df_patient2['cbg']


    df_patient1 = pd.DataFrame(imputer.transform(df_patient1), columns = columns)
    df_patient2 = pd.DataFrame(imputer.transform(df_patient2), columns = columns)


    df_patient1['cbg'] = cbg_pat1
    df_patient2['cbg'] = cbg_pat2
    del df_patient1['id']
    del df_patient2['id']


    return df_patient1, df_patient2



def preprocess_train(df_patient1, df_patient2):

    df_patient1['id'] = 0
    df_patient2['id'] = 1

    # Sorting based on time feature
    df_patient1 = df_patient1.sort_values(by='5minute_intervals_timestamp')
    df_patient2 = df_patient2.sort_values(by='5minute_intervals_timestamp')

    # Outlier removal
    df_patient1 = outlier_removal(df_patient1)
    df_patient2 = outlier_removal(df_patient2)


    # first imputation
    df_patient1 = first_imputation(df_patient1)
    df_patient2 = first_imputation(df_patient2)



    # second imputation and scaling
    imputer = KNNImputer(n_neighbors=3, weights="uniform")
    scaler = MinMaxScaler()

    df_patient1, df_patient2 = second_imputation_and_scaling(df_patient1, df_patient2, imputer, scaler)
    del df_patient1['id']
    del df_patient2['id']

    # save scaler and imputer for test set
    from constants import SAVE_PATH
    joblib.dump(scaler, SAVE_PATH + 'scaler_model.joblib')
    joblib.dump(imputer, SAVE_PATH + 'imputer_model.joblib')

    return df_patient1, df_patient2


def create_train_dataloaders(df, input_size, batch_size):

    dataset = TimeSeriesDataset(df, input_size)

    # Define the sizes for the training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # Use random_split to split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
