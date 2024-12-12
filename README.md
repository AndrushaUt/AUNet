# Audio Source Separation with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains a template for solving ASR task with PyTorch. This template branch is a part of the [HSE DLA course](https://github.com/markovka17/dla) ASR homework. Some parts of the code are missing (or do not follow the most optimal design choices...) and students are required to fill these parts themselves (as well as writing their own models, etc.).

See the task assignment [here](https://github.com/markovka17/dla/tree/2024/hw1_asr).

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Use

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

If you want to fine-tune our best model, you should download weights of our checkpoint. To do this, run the following command:
```bash
python3 download_weights.py vars.to_save="dir_name for saving best_model.pth"
```
Checkpoint will be saved in `path/to/download_weights.py/../saved/to_save_from_download.yaml/best_model.pth`

To run inference (evaluate the model or save predictions):

```bash
python3 inference.py inferencer.from_pretrained="path to your checkpoint" datasets.val.mix_audio_dir="Your path to mix audio dir" datasets.val.s1_audio_dir="Your path to s1 audio dir" datasets.val.s2_audio_dir="Your path to s2 audio dir"  
```

If you do not have ground truth, you can set null for s2_audio_dir and s1_audio_dir

## How to Calc Metrics
```python3
python3 calculate_metrics.py paths.s1_estimated_path="Your path for s1 estimated wavs" paths.s2_estimated_path="Your path for s2 estimated wavs" paths.s1_target_path="Your path for s1 target wavs" paths.s2_target_path="Your path for s2 target wavs" paths.mix_path="Your path for mix wavs" 
```

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
