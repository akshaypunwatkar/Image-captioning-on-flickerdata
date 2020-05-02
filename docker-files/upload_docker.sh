
#!/usr/bin/env bash
# This tags and uploads an image to Docker Hub


dockerpath="akshaypunwatkar/logoclassifier"

# Authenticate & Tag
echo "Docker ID and Image: $dockerpath"
docker login &&\
    docker image tag logo_clasifier $dockerpath

# Push Image
docker image push $dockerpath 
