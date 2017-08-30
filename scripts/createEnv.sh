ENV_PATH=$1
ENV_NAME=$2

cd $ENV_PATH; pyvenv $ENV_NAME
#python3 -m venv /path/to/new/virtual/environment
