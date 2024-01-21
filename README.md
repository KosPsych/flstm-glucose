# Predicting glucose levels over time using a federated learning framework incorporating LSTM and attention mechanisms
This repository hosts code for predicting glucose levels in a federated learning context using the [Flower](https://github.com/adap/flower) framework. The dataset employed is [OHIOT1DM](http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html), and it is available upon request from the authors. The current configuration of the repository accommodates an arbitrary number of clients, provided that the folders are appropriately configured. Users have the flexibility to adjust various parameters, such as the number of local or global training rounds and the learning rate, to conduct experiments and explore different settings. The code also includes an exploratory data analysis (EDA) regarding missing values and outliers.

------
## Set up

### Step 1: Cloning the repository
```
git clone https://github.com/KosPsych/flstm-glucose
```
### Step 2: Setting up folder structure

```
flstm-glucose/src/..
             /models/..
             /data/
               train/
                     0/
                     1/
                     2/
                    ...
               test/
                     0/
                     1/
                     2/
                    ...
```


