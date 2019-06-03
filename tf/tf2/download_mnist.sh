#!/usr/bin/env bash

DS_PATH="$1/"
if [ -d "$1" ]; then
  echo "WARNING: directory $DS_PATH already exists."
  exit
fi
echo "Downloading dataset into $DS_PATH"
DS_URL="http://yann.lecun.com/exdb/mnist/"
FILES=(
    "train-images-idx3-ubyte.gz"
    "train-labels-idx1-ubyte.gz"
    "t10k-images-idx3-ubyte.gz"
    "t10k-labels-idx1-ubyte.gz"
)
mkdir data/mnist || true
for filename in "${FILES[@]}"
do
    wget -P $DS_PATH $DS_URL$filename
done
