# Install notes
## Steps

1. Install below deps




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


