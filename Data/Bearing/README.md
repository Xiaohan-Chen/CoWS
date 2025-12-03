#### 1. working condition:
| Code  | Working condition |
| ----- | ----------------- |
| load0 | N15_M07_F10       |
| load1 | N09_M07_F10       |
| load2 | N15_M01_F10       |
| load3 | N15_M07_F04       |

#### 2. Lable:
| Label | Bearing code | Fault        | Location | Combination | Extend of damage | Characteristic |
| ----- | ------------ | ------------ | -------- | ----------- | ---------------- | -------------- |
| 0     | k002         | /            | /        | /           | /                | /              |
| 1     | KA15         | indentations | OR       | S           | 1                | single point   |
| 2     | KA16         | pitting      | OR       | R           | 2                | single point   |
| 3     | KA22         | pitting      | OR       | S           | 1                | single point   |
| 4     | KA30         | indentations | OR       | R           | 1                | distributed    |
| 5     | KB23         | pitting      | IR+OR    | M           | 2                | single point   |
| 6     | KB24         | pitting      | IR+OR    | M           | 3                | distributed    |
| 7     | KB27         | indentations | IR+OR    | M           | 1                | distributed    |
| 8     | KI14         | pitting      | IR       | M           | 1                | single point   |
| 9     | KI16         | pitting      | IR       | S           | 3                | single point   |
| 10    | KI17         | pitting      | IR       | R           | 1                | single point   |
| 11    | KI18         | pitting      | IR       | S           | 2                | single point   |
| 12    | KI21         | pitting      | IR       | S           | 1                | single point   |
OR: outer race; IR: inner race  
S: single damage; R: repetitive damage; M: multiple damage  

#### 3. Data structure of each file:
```
{
    label 0: numpy.ndarray # [number of sequence, sequence length]
    label 1: numpy.ndarray
    ...
    label 12: numpy.ndarray
}
```

#### 4. How to load data
```python
>>> import pickle
>>> with open('load0.pkl', 'rb') as ff:
        data = pickle.load(ff)
>>> print(data[0].shape)
(490, 1024)
```
This means: in load 0, label 0 contains 490 sequences with a length of 1024.


**NOTE**: All data have been normalized between 0 and 1.

#### 5. Citation
If you use the bearing dataset, please cite the following paper:
```
@inproceedings{lessmeier2016condition,
  title={Condition monitoring of bearing damage in electromechanical drive systems by using motor current signals of electric motors: A benchmark data set for data-driven classification},
  author={Lessmeier, Christian and Kimotho, James Kuria and Zimmer, Detmar and Sextro, Walter},
  booktitle={PHM Society European Conference},
  volume={3},
  number={1},
  year={2016}
}
```