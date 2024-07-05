# Learning spatial interaction representation with heterogeneous graph convolutional networks for urban land-use inference

Pytorch implementation of our paper "Learning spatial interaction representation with heterogeneous graph convolutional networks for urban land-use inference". We propose a novel framework—a heterogeneous graph convolutional network (HGCN)—to explicitly account for the spatial demand and supply components embedded in spatial interaction data. The HGCN can distinguish heterogeneous mechanisms in supply- and demand-related modalities of spatial interactions, incorporating both spatial interaction and spatial dependence for urban land-use inference.

## 1. Requirements
- torch
- numpy
- pandas
- gensim
- sklearn
- geopandas

## 2. Structure

Due to commercial and legal restrictions in China, we are unable to share certain data. Instead, we provide mock data to demonstrate how the codes work.

| Folder/File                            | Content                                   |
| :------------------------------------- | :---------------------------------------- |
| ./mock_data/{SZ/LD}_data/              | Mock data for Shenzhen/London.            |
| ./mock_data/{SZ/LD}_data_process.ipynb | Code for processing Shenzhen/London data. |
| {SZ/LD}_train.py                       | Code for training the HGCN model.         |

## 3. Commands

You can train the HGCN model in the Shenzhen/London dataset according to the following instructions.

```bash
############################### Shenzhen ################################
python SZ_train.py --seed 30 --epochs 400 --k-fold 30 --train_size 0.7
###############################  London  ################################
python LD_train.py --seed 30 --epochs 400 --k-fold 30 --train_size 0.7
```

## 4. Data links

London: https://pan.baidu.com/s/19d-8LKsgDmoY45XSeOHIvA (vrw8)

Shenzhen: https://pan.baidu.com/s/1eQ12lWjHrT2D8o66WTP6FA (wf0t)
