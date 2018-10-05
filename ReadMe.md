# Vanilla NER
<!-- 
[![Documentation Status](https://readthedocs.org/projects/ld-net/badge/?version=latest)](http://ld-net.readthedocs.io/en/latest/?badge=latest) -->
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This project is drivied from LD-Net, and provides a vanilla Char-LSTM-CRF model for Named Entity Recognition. 

We are in an early-release beta. Expect some adventures and rough edges. LD-Net is a more mature project, please refer to LD-Net for detailed documents and also demo scripts.

https://github.com/LiyuanLucasLiu/LD-Net


## Training

### Dependency

Our package is based on Python 3.6 and the following packages:
```
numpy
tqdm
torch-scope
torch==0.4.1
```

### Command

Please first generate the word dictionary by:
```
python pre_seq/gene_map.py -h
```

Then encode the dictionary by:
```
python pre_seq/encode_data.py -h
```

Then train the model:
```
python train_seq.py -h
```

## Citation

If you find the implementation useful, please cite the following paper: [Efficient Contextualized Representation: Language Model Pruning for Sequence Labeling](https://arxiv.org/abs/1804.07827)
```
@inproceedings{liu2018efficient,
  title = "{Efficient Contextualized Representation: Language Model Pruning for Sequence Labeling}", 
  author = {Liu, Liyuan and Ren, Xiang and Shang, Jingbo and Peng, Jian and Han, Jiawei}, 
  booktitle = {EMNLP}, 
  year = 2018, 
}
```
