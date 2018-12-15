# Vanilla NER
<!-- 
[![Documentation Status](https://readthedocs.org/projects/ld-net/badge/?version=latest)](http://ld-net.readthedocs.io/en/latest/?badge=latest) -->
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Check Our New NER Toolkit🚀🚀🚀**
- **Inference**:
  - **[LightNER](https://github.com/LiyuanLucasLiu/LightNER)**: inference w. models pre-trained / trained w. *any* following tools, *efficiently*. 
- **Training**:
  - **[LD-Net](https://github.com/LiyuanLucasLiu/LD-Net)**: train NER models w. efficient contextualized representations.
  - **[VanillaNER](https://github.com/LiyuanLucasLiu/Vanilla_NER)**: train vanilla NER models w. pre-trained embedding.
- **Distant Training**:
  - **[AutoNER](https://shangjingbo1226.github.io/AutoNER/)**: train NER models w.o. line-by-line annotations and get competitive performance.

--------------------------------

This project is drivied from LD-Net, and provides a vanilla Char-LSTM-CRF model for Named Entity Recognition (LD-Net w.o. contextualized representations). 

We are in an early-release beta. Expect some adventures and rough edges. LD-Net is a more mature project, please refer to LD-Net for detailed documents and also demo scripts.

https://github.com/LiyuanLucasLiu/LD-Net


## Training

### Dependency

Our package is based on Python 3.6 and the following packages:
```
numpy
tqdm
torch-scope>=0.5.0
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

## Inference

Models trained with this package can be used to inference with the [LightNER package](https://github.com/LiyuanLucasLiu/LightNER).

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
