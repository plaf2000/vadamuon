# Install

The environment used for running the experiment was created using [uv](https://docs.astral.sh/uv/). We recommend using this setting.

## With uv

In case uv is installed, run the following:
```
uv sync
uv pip install -e .
```
The `-e` flag makes it possible to modify the library, and modifications will be loaded on the fly.

## With pip

In case uv is not available, we still provide a file `requirements.txt` to install the dependencies using pip, although this has not been tested. In a virtual environment, run:

```
pip install -r requirements.txt
pip install -e .
```
The `-e` flag makes it possible to modify the library, and modifications will be loaded on the fly.


# The optimizers

Inside the [vadam](./vadam) folder is implemented the VadaMuon optimizer. Note that the implementation contains a variant that uses some RMS regularizarion. This was not presented in the report as it showed worse performance.

# Running the Experiments

## MNIST

Please refer to the [mnist_code](./mnist_code/) folder for the details concerning the experiments you can run on this dataset.

## Others

Other experiment are not implemented for AdaMuon and the code is left unchanged from the original repository.
