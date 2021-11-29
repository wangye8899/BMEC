## Overview

This repo contains BMEC dataset and related model code.

> code -----> model and train code
>
> data ------> BMEC dataset
>
> results ------> experiments results

---

## Results

**Loss curve**

![](https://github.com/wangye8899/BMEC/blob/main/results/loss.png)

**Accuracy curve**

![](https://github.com/wangye8899/BMEC/blob/main/results/acc.png)

---

## How to train

Firstly, you should  create the same env according to the `environment.yml`

```bash
conda env create -f environment.yml
```

Secondly, you should unzip the BMEC dataset in `data` folder

```bash
cd data
unzip red_4_cells_raw.zip
```

Thirdly, you may revise the data path in `train.py`

```python
parser.add_argument('--data_dir', metavar='DIR', default="your data path",
                    help='path to dataset')
```

Finally, you just run `train.py` to finish the training process.

```bash
nohup python train.py  > ./Log/Rotate/swin_transformer.log 2>&1 &
```

---

## Others

Our Shape Attention module is implemented in `shape_attention.py`

Classification Network is implemented in `classification_net.py`





