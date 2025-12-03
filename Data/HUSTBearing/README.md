#### 1. working condition:
| Code  | Working condition |
| ----- | ----------------- |
| load0 | 65Hz              |
| load1 | 70Hz              |
| load2 | 75Hz              |
| load3 | 80Hz              |

#### 2. Lable:
| Label | Bearing code  |
| ----- | ------------- |
| 0     | normal        |
| 1     | inner race    |
| 2     | outer race    |
| 3     | ball          |
| 4     | combination   |

#### 3. Data structure of each file:
```
{
    label 0: numpy.ndarray # [number of sequence, sequence length]
    label 1: numpy.ndarray
    ...
    label 4: numpy.ndarray
}
```

#### 4. How to load data
```python
>>> import pickle
>>> with open('load0.pkl', 'rb') as ff:
        data = pickle.load(ff)
>>> print(data[0].shape)
(254, 1024)
```
This means: in load 0, label 0 contains 254 sequences with a length of 1024.

X-axis vibration data are used in this dataset.

**NOTE**: All data have been normalized between 0 and 1.

#### 5. Citation

```
Chao Zhao, Enrico Zio, Weiming Shen, Domain Generalization for Cross-Domain Fault
Diagnosis: an Application-oriented Perspective and a Benchmark Study, Reliability Engineering
and System Safety (2024), doi: https://doi.org/10.1016/j.ress.2024.109964.
```