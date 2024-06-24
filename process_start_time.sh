#!/bin/bash

# Get the list of PIDs from nvidia-smi
pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)

# Loop through each PID and get the start time using ps
echo "GPU Processes and their start times:"
for pid in $pids; do
    start_time=$(ps -p $pid -o lstart=)
    echo "PID $pid started at: $start_time"
done