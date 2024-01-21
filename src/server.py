from typing import List, Tuple
import numpy as np
import flwr as fl
from flwr.common import Metrics
import argparse

# Define metric aggregation function
def average_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    
    mse = [i[1]['MSE'] for i in metrics]
    rmse = [i[1]['RMSE'] for i in metrics]
    # Aggregate and return custom metric (weighted average)
    return {"MSE": np.mean(mse), "RMSE": np.mean(rmse)}



parser = argparse.ArgumentParser(description= "Flower")
parser.add_argument(
    "--n_clients",
    required=True,
    type=int,
    help="Number of federated learning clients.",
)
n_clients = parser.parse_args().n_clients



# Define strategy
strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn = average_metrics,
    min_fit_clients = n_clients,
    min_evaluate_clients = n_clients,
    min_available_clients = n_clients
    )


# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds = 3),
    strategy=strategy,
)
