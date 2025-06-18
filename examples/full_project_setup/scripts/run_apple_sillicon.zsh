#!/usr/bin/env zsh

set -xe

docker build --platform linux/amd64 --tag learning_machines .

docker run -t --rm \
  --platform linux/amd64 \
  -p 45100:45100 -p 45101:45101 \
  -v "$(pwd)/results:/root/results" \
  -e SAVE_DIR=/root/results \
  learning_machines "$@"

# Because docker runs as root, this means the files will be owned by the root user.
# Change this with:
# sudo chown "$USER":"$USER" ./results -R