# Install notes
## Steps

1. Install below deps

2. 
```sh
poetry config virtualenvs.prefer-active-python 
poetry config virtualenvs.prefer-active-python true
pyenv install 3.10.15
pyenv local 3.10.15  # Activate Python 3.10 for the current project # https://github.com/pyenv/pyenv/blob/master/COMMANDS.md#pyenv-local
python --version # to check current version, make sure its 3.10.15
poetry install

poetry install -E mujoco

# for newer GPUs:
poetry run pip install torch==1.12.1 --upgrade --extra-index-url https://download.pytorch.org/whl/cu113


# To fix buggy install:
poetry run pip install --force-reinstall torch==1.12.1 --upgrade --extra-index-url https://download.pytorch.org/whl/cu113

# To remove:
poetry env remove $(which python)
```



## Required deps
### Poetry
[[Python Poetry]]

#### Installing Poetry
Seth says to install with official installer for some reason:
<https://python-poetry.org/docs/#installing-with-the-official-installer>


```
curl -sSL https://install.python-poetry.org | python3 -
```

Installs poetry to:
```
$HOME/.local/bin
```
So we need to add it to path:
```
export PATH="$HOME/.local/bin:$PATH"
```


#### Using Poetry
<https://python-poetry.org/docs/basic-usage/>

Its kinda just like a venv wrapper:
```
poetry install

poetry shell # activate function

python hello_world.py

```
Need to check how to use debugger
#### Using Debugger / Interpreter Path with Poetry:
<https://github.com/python-poetry/poetry/issues/106>
```
poetry show -v
```


poetry env remove 
