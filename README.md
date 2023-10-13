# Text Lab

## Overview

Text Lab is a public template for artificial intelligence and machine learning research projects using Lightning AI's [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/) to train a Transformer model that is written in PyTorch.

The recommended way for Text Lab users to create new repos is with the [use this template](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template) button.

## Source Module

`textlab.core` should contain code for the Lightning Module and Trainer.

`textlab.components` should contain experiment utilities grouped by purpose for cohesion.

`textlab.pipeline` should contain code for data acquistion and preprocessing, and building a TorchDataset and LightningDataModule.

`textlab.serve` should contain code for model serving APIs built with [FastAPI](https://fastapi.tiangolo.com/project-generation/#machine-learning-models-with-spacy-and-fastapi).

`textlab.cli` should contain code for the command line interface built with [Typer](https://typer.tiangolo.com/)and [Rich](https://rich.readthedocs.io/en/stable/).

`textlab.config` can assist with project, trainer, and sweep configurations.

## Base Requirements and Extras

Text Lab installs minimal requirements out of the box, and provides extras to make creating robust virtual environments easier. To view the requirements, in [setup.cfg](setup.cfg), see `install_requires` for the base requirements and `options.extras_require` for the available extras.

The recommended install is as follows:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```
