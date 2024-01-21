#!/bin/bash
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 n_clients"
    exit 1
fi

n_clients=$1

python src/server.py  --n_clients=$n_clients&
sleep 10 

((n_clients = n_clients - 1))

for i in `seq 0 $n_clients`; do
    echo "Starting client $i"
    python src/client.py --node-id=${i}  &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait