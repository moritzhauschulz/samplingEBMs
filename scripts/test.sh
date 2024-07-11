#!/bin/bash

# Example path
path="/vol/bitbucket/meh23/samplingEBMs/methods/dataq_dfs/experiments/2spirals/2spirals_13/ckpts/model_100000.pt"

# Extract the file name using parameter expansion
filename="${path##*/}"

# Print the file name
echo "The file name is: $filename"
