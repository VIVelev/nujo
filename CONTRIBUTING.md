# Contributing to nujo

First off, thank you for considering contributing to nujo. It's people like you that make nujo such a great tool.

## Reporting issues
  - Describe what you expect to happen
  - Describe what happened
  - If you could, include a [minimal reproducible example](https://stackoverflow.com/help/minimal-reproducible-example)
  - List vesions of your Python, NumPy, Requests and GarphViz

## Submiting code
  - It is recommended to use [Visual Studio Code](https://code.visualstudio.com/)
  - All commits will be tested against flake8 linter, so make sure to abide by PEP8

### Commit messages
  - "[up:*branch*] *commit-message*" - creating bugs
  - "[fix:*branch*] *commit-message*" - correcting bugs
  - "[rm:*branch*] *commit-message*" - delete files
  - For the branch name, skip the type(feature, release, hotfix)

### Branching strategy
We are using *Gitflow* as a branching strategy.
We have the following branches:
  - **master** - stable, production ready code; latest stable release
  - **develop** - main branch for developing, all other branches get merged here
  - **feature/feature-name** - implementation of specific features
  - **release/release-tag** - release cycle; a release is made out of the **develop** branch
  - **hotfix/hotfix-name** - fixing important bugs in releases quickly

[Gitflow reference](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow)

### Task Management Tool
The Kanban tool used for this project is [Trello](https://trello.com/b/fObyuiWt/nujo-develop)

## Getting started

### Prerequisites

-   [Python](https://www.python.org/) - The Programming Language used (version 3.6+)
-   [Poetry](https://python-poetry.org/) - Dependency and Virtual Environment Management

***Download for Mac OSX using Homebrew***

```bash
$ brew install python poetry
```

### Installing and setting up nujo

Run the following in the terminal:
```bash
$ git clone https://github.com/VIVelev/nujo && cd nujo
$ poetry install && poetry shell
```

## Running the tests

Once you have **Installed and set up** nujo, run:

```bash
$ pytest
```

## Your First Contribution
>coming soon...

### When and how to write unit tests?
>coming soon...

### Pyhton Docstrings format
Use the following format to document python code.

```
''' Name of the Python object you are documenting

Short description

Parameters:
-----------
 - param_name : param type, description...

Returns:
--------
 - return_name : return type, description...

'''
```
