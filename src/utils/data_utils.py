from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import KNNImputer, IterativeImputer
from dataset import *
from torch.utils.data import random_split
import joblib
import os

def get_all_files_in_folder(folder_path):
    """
    Get a list of all files in the specified folder and its subfolders.

    Args:
        folder_path (str): Path to the folder.

    Returns:
        list: List of file paths.
    """
    all_files = []

    # Walk through the directory and get all files
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)

    return all_files





def outlier_removal(df):
    """
    Remove outliers from specific columns in the DataFrame.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: DataFrame with outliers removed.
    """
    # Set a maximum value for 'gsr' column
    df['gsr'] = df['gsr'].apply(lambda x: min(x, 25))

    # Set a maximum value for 'carbInput' column
    df['carbInput'] = df['carbInput'].apply(lambda x: min(x, 250))

    return df



def first_imputation(df):
    """
    Perform initial imputation on the DataFrame.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: DataFrame with imputed values.
    """
    # Impute missing values in 'carbInput', 'bolus', and 'basal' columns using forward fill
    df['carbInput'] = df['carbInput'].fillna(method='ffill').fillna(0)
    df['bolus'] = df['bolus'].fillna(method='ffill').fillna(0)
    df['basal'] = df['basal'].fillna(method='ffill').fillna(0)

    # Impute 'cbg' values based on 'finger' when 'missing_cbg' is 1
    df.loc[df['missing_cbg'] == 1, 'cbg'] = df['finger']

    # Drop unnecessary columns
    df.drop(columns=['finger', 'hr', 'missing_cbg', '5minute_intervals_timestamp'], inplace=True)

    return df



def second_imputation_and_scaling(dataframes, imputer, scaler):
    """
    Perform a second imputation and scaling on a list of DataFrames.

    Args:
        dataframes (list): List of input DataFrames.
        imputer: Imputer object for filling missing values.
        scaler: Scaler object for scaling values.

    Returns:
        list: List of imputed and scaled DataFrames.
    """
    # Concatenate all DataFrames in the list
    df = pd.concat(dataframes)

    # Scale the concatenated DataFrame
    scaled_df = scaler.fit_transform(df)

    # Impute missing values in the scaled DataFrame
    imputed_df = pd.DataFrame(imputer.fit_transform(scaled_df), columns=df.columns)

    # Split the imputed DataFrame back into a list of DataFrames
    return [group.drop('id', axis=1) for _, group in imputed_df.groupby('id')]




def preprocess_test(dataframes):
    """
    Preprocess a list of test DataFrames.

    Args:
        dataframes (list): List of input DataFrames.

    Returns:
        list: List of preprocessed DataFrames.
    """
    # Get the length of the list of DataFrames
    length = len(dataframes)

    from utils.constants import MODEL_PATH
    # Load pre-trained imputer and scaler models
    imputer = joblib.load(MODEL_PATH + 'imputer_model.joblib')
    scaler = joblib.load(MODEL_PATH + 'scaler_model.joblib')

    # Iterate through each DataFrame in the list
    for i in range(length):
        # Sort the DataFrame based on the '5minute_intervals_timestamp' feature
        dataframes[i] = dataframes[i].sort_values(by='5minute_intervals_timestamp')
        
     
        
        # Apply the first imputation to the DataFrame
        dataframes[i] = first_imputation(dataframes[i])
        
        # Add an 'id' column to identify the source DataFrame
        dataframes[i]['id'] = i
        
        # Get the column names before scaling
        columns = dataframes[i].columns
        
        # Scale the DataFrame using the pre-trained scaler
        dataframes[i] = pd.DataFrame(scaler.transform(dataframes[i]), columns=columns)
        
        # Save the 'cbg' column separately before imputation
        cbg = dataframes[i]['cbg']
        
        # Apply the imputation to the DataFrame
        dataframes[i] = pd.DataFrame(imputer.transform(dataframes[i]), columns=columns)
        
        # Remove the 'id' column
        del dataframes[i]['id']

    return dataframes



def preprocess_train(dataframes):
    """
    Preprocess a list of training DataFrames.

    Args:
        dataframes (list): List of input DataFrames.

    Returns:
        list: List of preprocessed and scaled DataFrames.
    """
    # Get the length of the list of DataFrames
    length = len(dataframes)

    # Iterate through each DataFrame in the list
    for i in range(length):
        # Sort the DataFrame based on the '5minute_intervals_timestamp' feature
        dataframes[i] = dataframes[i].sort_values(by='5minute_intervals_timestamp')
        
        # Apply outlier removal to the DataFrame
        dataframes[i] = outlier_removal(dataframes[i])
        
        # Apply the first imputation to the DataFrame
        dataframes[i] = first_imputation(dataframes[i])
        
        # Add an 'id' column to identify the source DataFrame
        dataframes[i]['id'] = i

    # Initialize an IterativeImputer and MinMaxScaler for second imputation and scaling
    imputer = IterativeImputer()
    scaler = MinMaxScaler()

    # Apply second imputation and scaling to the list of DataFrames
    dataframes = second_imputation_and_scaling(dataframes, imputer, scaler)

    # Save the scaler and imputer models for the test set
    from utils.constants import MODEL_PATH
    joblib.dump(scaler, MODEL_PATH + 'scaler_model.joblib')
    joblib.dump(imputer, MODEL_PATH + 'imputer_model.joblib')

    return dataframes


def create_train_dataloaders(datasets, input_size, batch_size):
    """
    Create training and validation data loaders for a list of datasets.

    Args:
        datasets (list): List of TimeSeriesDataset instances.
        input_size (int): Size of the input sequences.
        batch_size (int): Batch size for the data loaders.

    Returns:
        tuple: Tuple containing lists of training and validation data loaders.
    """
    val_loaders = []
    train_loaders = []

    # Iterate through each dataset in the list
    for df in datasets:
        # Create a TimeSeriesDataset instance for the current dataset
        dataset = TimeSeriesDataset(df, input_size)

        # Define the sizes for the training and validation sets
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        # Use random_split to split the dataset into training and validation sets
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Create DataLoader instances for training and validation sets
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Append the data loaders to the respective lists
        val_loaders.append(val_loader)
        train_loaders.append(train_loader)

    return train_loaders, val_loaders
