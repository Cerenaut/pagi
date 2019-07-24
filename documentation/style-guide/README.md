# Project AGI Style Guide

## Summary
Any code in Python and TensorFlow should adhere to the [Project AGI Coding Principles](./principles.md), [TensorFlow Style Guide](https://www.tensorflow.org/community/style_guide), and
any additional modifications outlined below.

### Rule Modifications
- Maximum characters per line is **120 characters**, instead of 80.

## Quick Start
### Installation
1. Install the [Pylint](https://www.pylint.org) plugin using `pip install pylint`
2. Download the [pylintrc](./pylintrc) file and place it in your home directory as `.pylintrc`
    - Read [this section](http://pylint.pycqa.org/en/latest/user_guide/run.html?highlight=pylintrc#command-line-options) of the 
    documentation to understand how `pylint` files the correct rules configuration
3. Verify `pylint` is using the correct rules configuration by running `pylint .` in any Python project

#### Quick Example
```
› pylint .
Using config file /Users/Abdel/.pylintrc
*************
F:  1, 0: error while code parsing: Unable to load file __init__.py:
[Errno 2] No such file or directory: '__init__.py' (parse-error)
```

### Editor/IDE Integration
Refer to the [Editor and IDE integration](https://pylint.readthedocs.io/en/latest/user_guide/ide-integration.html) section in the 
Pylint documentation for more information about other editors.

- Visual Studio Code: integrated by default, no need to install plugins.
- Sublime Text: Install the [SublimeLinter-pylint](https://github.com/SublimeLinter/SublimeLinter-pylint) after following the 
instructions above to install `pylint`
- PyCharm: No easy plugin available due to PyCharm's default linting. Follow these [instructions](https://pylint.readthedocs.io/en/latest/user_guide/ide-integration.html#pylint-in-pycharm) 
from the Pylint documentation to integrate with pylint

## Pre-commmit Hooks
We use the `pre-commit` package alongside the config file `.pre-commit-config.yaml` to enforce styling rules on newly committed code. The package should be installed automatically as a dependencing when you first install `pagi`. You then need to run `pre-commit install` to install the hooks in your local Git repository.

### Command-line Usage
You can also use `pylint` using the command-line, instead of integrating with your editor/IDE of choice.

```
# Run pylint on files in the all directories within the project
pylint **/*.py

# Run pylint on a specific Python module (must contain __init__.py)
pylint parent_module/child_module

# Optionally pass a custom rcfile
pylint --rcfile=/path/to/pylintrc **/*.py
```
