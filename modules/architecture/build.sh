function cleanImages() {

    docker rmi -f "$(docker images | grep -m 1 'frontend' | awk '{print $3}')"

}

function cleanVolumes() {
  docker volume rm "frontend"
}

function buildImages() {

  docker build \
      --build-arg DOMAIN=http://localhost:8081 \
    -f deployment/images/frontend/Dockerfile \
    -t frontend .

}

function dockerCompose() {

    docker-compose up

}

# ----------------------------------------------------------------------------------------------------------------------
# -- Main --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

cleanImages;
cleanVolumes;
buildImages;
dockerCompose;
