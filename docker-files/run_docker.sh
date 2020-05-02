#!/usr/bin/env bash

# Build image
docker build --tag=logo_clasifier .

# List docker images
docker image ls

# Run flask app
docker run -p 8088:8088 logo_clasifier
