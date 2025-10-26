# Installation with Docker (or any OCI container engine)

## More details on the setup

The setup is based on Docker and Docker Compose and is adapted from
the [Cresset template](https://github.com/cresset-template/cresset).
It is composed of Dockerfiles to build the image containing the runtime environment,
and Docker Compose files to set build arguments in the Dockerfile and run it locally.

Here's a summary of all the files in this directory.

```
docker-amd64-cuda-vllm/
├── Dockerfile                       # Dockerfile template. Edit if you are building things manually.
├── Dockerfile-user                  # Dockerfile template. Adds the dev and user layers.
├── compose-base.yaml                # Sets the build args for the Dockerfile.
│                                    # Edit to change the base image or package manager.
├── compose.yaml                     # Docker Compose template. Edit if you have a custom local deployment or change the hardware acceleration.
├── template.sh                      # A utility script to help you interact with the template (build, deploy, etc.).
├── .env                             # Will contain your personal configuration. Edit to specify your personal configuration.
├── environment.yml                  # If chose the `from-scratch` option. Conda and pip dependencies.
├── requirements.txt                 # If chose the `from-python` option. pip dependencies.
├── apt.txt                          # Apt dependencies ("system" dependencies).
├── update-env-file.sh               # Template file. A utility script to update the environment files.
└── entrypoints/
    ├── entrypoint.sh                # The main entrypoint that install the project and triggers other entrypoints.
    ├── pre-entrypoint.sh            # Runs the base entrypoint of the base image if it has one.
    ├── logins-setup.sh              # Manages logging into services like wandb.
    └── remote-development-setup.sh  # Contains utilities for setting up remote development with VSCode, PyCharm, Jupyter.

```

### Details on the main Dockerfile

The Dockerfile specifies all the steps to build the environment in which your code will run.
It uses the files `apt.txt`, `requirements.txt`, and `environment.yml` to install the system and Python dependencies.
You typically don't have to edit the Dockerfile directly, unless you need to install dependencies manually
(not as a one time install from the dependencies files).
The `Dockefile-user` is used to build the user layer on top of the generic image for OCI runtimes with limitations
on user creation at container creation.

### Details on the Docker Compose files

The Docker Compose files are used to configure the build arguments used by the Dockerfile
when building the images and to configure the container when running it locally.

They support building multiple images corresponding to the runtime and development stages with or without a user
and running on each with either `cpu` or `cuda` support.

We provide a utility script, `template.sh`, to help you interact with Docker Compose.
It has a function for each of the main operations you will have to do.

You can always interact directly with `docker` or `docker compose` if you prefer
and get examples from the `./template.sh` script.

### Prerequisites

* `docker` (A Docker Engine `docker version` >= v23). [Install here.](https://docs.docker.com/engine/)
* `docker compose` (`docker compose version` >= v2). [Install here.](https://docs.docker.com/compose/install/)

### Clone the repository

Clone the git repository.

```bash
git clone <HTTPS/SSH> packing
cd packing
```

### Obtain/build the images

All commands should be run from the `installation/docker-amd64-cuda-vllm/` directory.

```bash
cd installation/docker-amd64-cuda-vllm
```

1. Create an environment file for your personal configuration with
   ```bash
   ./template.sh env
   ```
   This creates a `.env` file with pre-filled values.
    - The `USRID` and `GRPID` are used to give the container user read/write access to the storage that will be mounted
      when the container is run with a container setup that does not change the user namespace
      (typically the case with rootful Docker and on the EPFL runai cluster).
      Edit them so that they match the user permissions on the mounted volumes, otherwise you can leave them as is.
      (If you're deploying locally, i.e., where you're building, these values should be filled correctly by default.)

2. Pull or build the generic image.
   This is the image with root as user.
   It will be named according to the image name in your `.env`.
   It will be tagged with `<platform>-root-latest` and if you're building it,
   it will also be tagged with the latest git commit hash `<platform>-root-<sha>` and `<platform>-root-<sha>`.
    - Pull the generic image if it's available.
      ```bash
      # Pull the generic image if available.
      ./template.sh pull_generic TODO ADD PULL_IMAGE_NAME (private or public).
      ```
    - Otherwise, build it.
      ```bash
      ./template.sh build_generic
      ```
3. You can run quick checks on the image to check it that it has what you expect it to have:
   ```bash
   # Check all your dependencies are there.
   ./template.sh list_env

    # Get a shell and check manually other things.
    # This will only contain the environment and not the project code.
    # Project code can be debugged on the cluster directly.
    ./template.sh empty_interactive
   ```

4. Build the image configured for your user. (Not needed to run in containers with new namespaces like for the SCITAS and CSCS clusters)
   ```bash
   ./template.sh build_user
   ```
   This will build a user layer on top of the generic image
   and tag it with `*-${USR}` instead of `*-root`.
   This will be the image that you run and deploy to match the permissions on your mounted storage in container
   setups that maintain the user namespace (e.g., rootful Docker, the EPFL runai cluster).

## Licenses and acknowledgements

This Docker setup is based on the [Cresset template](https://github.com/cresset-template/cresset)
with the LICENSE.cresset file included in this directory.
