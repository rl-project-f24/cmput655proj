# GPU setup

Basically, for 3070:
poetry run pip install --force-reinstall torch==2.5.1 --upgrade --extra-index-url https://download.pytorch.org/whl/cu124

nvidia-smi to see the cuda version in top right.



## Resetting env to try and fix 

poetry env remove $(which python)

poetry install -E mujoco


poetry run pip install torch==1.12.1 --upgrade --extra-index-url https://download.pytorch.org/whl/cu113


poetry shell



## Debugging

poetry run pip show torch




poetry run pip install --force-reinstall torch==1.12.1 --upgrade --extra-index-url https://download.pytorch.org/whl/cu113



poetry run pip install --force-reinstall torch==1.12.1 --upgrade --extra-index-url https://download.pytorch.org/whl/cu118


poetry run pip install --force-reinstall torch==2.5.1 --upgrade --extra-index-url https://download.pytorch.org/whl/cu124



## Found issue:
nvidia-smi shows my cuda version at 12.7, but the default installed version is for cuda version CUDA version: 11.3 (based on the +cu113 tag).



