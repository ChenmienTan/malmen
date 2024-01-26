# MALMEN: MAssive Language Model Editing Network

This is the official repo of our ICLR'24 paper [Massive Editing for Large Language Models via Meta Learning](https://arxiv.org/pdf/2311.04661.pdf).
You can email `chenmien.tan@ed.ac.uk` for any issue.

## Setup

You can create a virtual environment and install the dependencies via [Anaconda](https://www.anaconda.com).
```
$ conda create -n malmen
$ conda activate malmen
(malmen)$ pip install -r requirements.txt
```
The datasets for all experiments presented in the manuscript are available at [this Google Drive link](https://drive.google.com/drive/folders/1gu5tdk7MyL7tGWhITINN0YIi_qeTJAAe?usp=share_link).
You need to specify the paths to the `json` files in `config.data.train_path` and `config.data.valid_path`.
You should also specify an empty folder in `config.editor.cache_dir` to store cache files generated during running the code.

## Running

You can set all hyper-parameters via modifying the `yaml` files in the folder `config`.
You should run the code by executing the `main.py` file.
You can also specify the hyper-parameters on the command line.
```
(malmen)$ python main.py  \
    data=zsre  \
    model=gpt-j  \
    editor=malmen
```

## Acknowledgement

We thank the implementation of [MEND](https://github.com/eric-mitchell/mend) and [MEMIT](https://github.com/kmeng01/memit), which inspires some  code in this repo.

## Citation


```
@inproceedings{tan23malmen,
    title={Massive Editing for Large Language Models via Meta Learning},
    author={Chenmien Tan and Ge Zhang and Jie Fu},
    booktitle={International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/pdf?id=L6L1CJQ2PE}
}
```