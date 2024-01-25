# Federated Learning for Glucose Prediction with LSTM and Attention
This repository hosts code for predicting glucose levels in a federated learning context using the [Flower](https://github.com/adap/flower) framework. The dataset employed is [OHIOT1DM](http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html), and it is available upon request from the authors. The current configuration of the repository accommodates an arbitrary number of clients, provided that the folders are appropriately configured. Users have the flexibility to adjust various parameters, such as the number of local or global training rounds and the learning rate, to conduct experiments and explore different settings. The code also includes an exploratory data analysis (EDA) regarding missing values and outliers.

------
## Set up

### Step 1: Cloning the repository
```
git clone https://github.com/KosPsych/flstm-glucose
```
### Step 2: Setting up folder structure

Create a models and a data folder to handle saved models and the dataset respectively.
The structure of the data folder should adhere to the following format, wherein each of the subfolders labeled 0, 1, 2, and so forth, must contain the patient's CSV data for both training and testing.
```
flstm-glucose/src/..
             /models/..
             /data/
               train/
                     0/..
                     1/..
                     2/..
                    ...
               test/
                     0/..
                     1/..
                     2/..
                    ...
```
### Step 3: Add models and data folder path to ```constants.py```
### Step 4: Create a virtual environment with Python version 3.10

```
conda create --name <name> python==3.10
```

### Step 5: Install dependencies

```
pip install -r requirements.txt
```

----------------


## <u>Running</u>
If the preceding steps have been executed accurately, the code should be ready to run.

### Option 1
Open $n+1$ terminals where $n$ is the desired number of clients.
Then start the server and each client with:

```
python3 src/server.py --n_clients=n
python3 src/client.py --node-id=0
python3 src/client.py --node-id=1
...
python3 src/client.py --node-id=n-1
```
The federated learning process will start and you will be able to see local training losses in the clients terminals and aggregated test set metrics for each federated round in the server's terminal.

### Option 2
For $n$ clients, run:
```
bash run.sh n
```
This will do the same thing with Option 1 without opening $n+1$ terminals while displaying all losses and metrics in the same terminal.
