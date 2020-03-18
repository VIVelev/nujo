# Neural Dojo: A Reverse-mode Automatic Differentiation library for Neural Networks

[![lint_and_test_workflow](https://github.com/VIVelev/nujo/workflows/Lint%20and%20Test/badge.svg)](https://github.com/VIVelev/nujo/actions?query=workflow%3A%22Lint+and+Test%22)

### Prerequisites

-   [Python](https://www.python.org/) - The Programming Language used
-   [Poetry](https://python-poetry.org/) - Dependency and Virtual Environment Management

***Download for Mac OSX using Homebrew***

```bash
$ brew install python poetry
```

## Installing and setting up

### For users:
```bash
$ pip install nujo
```

Now run the following to make sure nujo was installed properly:
```bash
$ python                                                                                                                       
>>> import nujo as nj
>>> nj.__version__
'0.1.0'
>>> 
```

### For developers:

```bash
$ git clone https://github.com/VIVelev/nujo && cd nujo
$ poetry install && poetry shell
```

## Running the tests

Once you have **Installed and set up** Nujo, run the following in the terminal:

```bash
$ pytest
```

## Built With

-   [NumPy](http://www.numpy.org/) - Fundamental package for scientific computing with Python
-   [Graphviz](https://www.graphviz.org/) - Open source graph visualization software
-   [Requests](https://requests.readthedocs.io/en/master/) - HTTP for Humans

## Contributing

Please read [CONTRIBUTING.md](https://github.com/VIVelev/nujo/blob/master/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

-   **Victor Velev** - _Initial work_ - [VIVelev](https://github.com/VIVelev)
-   **Victor Gorchilov** - _Initial work_ - [ManiacMaxo](https://github.com/ManiacMaxo)

See also the list of [contributors](https://github.com/VIVelev/nujo/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
