#### 1. working condition:
| Code  | Working condition |
| ----- | ----------------- |
| load0 | 35Hz12kN          |
| load1 | 37.5Hz11kN        |
| load2 | 40Hz10kN          |

#### 2. Lable:
| Load  | Label | Bearing code  |
| ----- | ----- | ------------- |
| load0 | 0     | normal        |
|       | 1     | outer race    |
|       | 2     | cage          |
|       | 3     | mixed         |
| load1 | 0     | normal        |
|       | 1     | inner race    |
|       | 2     | outer race    |
|       | 3     | cage          |
| load2 | 0     | normal        |
|       | 1     | inner race    |
|       | 2     | outer race    |
|       | 3     | mixed         |


#### 3. Data structure of each file:
```
{
    label 0: numpy.ndarray # [number of sequence, sequence length]
    label 1: numpy.ndarray
    ...
    label 3: numpy.ndarray
}
```

#### 4. How to load data
```python
>>> import pickle
>>> with open('load0.pkl', 'rb') as ff:
        data = pickle.load(ff)
>>> print(data[0].shape)
(240, 1024)
```
This means: in load 0, label 0 contains 254 sequences with a length of 1024.

**NOTE**: All data have been normalized between 0 and 1.

#### 5. Citation

```
雷亚国, 韩天宇, 王彪, 李乃鹏, 闫涛, 杨军. XJTU-SY滚动轴承加速寿命试验数据集解读[J]. 机械工程学报, 2019, 55(16): 1-6 https://doi.org/10.3901/JME.2019.16.001

LEI Yaguo, HAN Tianyu, WANG Biao, LI Naipeng, YAN Tao, YANG Jun. XJTU-SY Rolling Element Bearing Accelerated Life Test Datasets: A Tutorial[J]. Journal of Mechanical Engineering, 2019, 55(16): 1-6 https://doi.org/10.3901/JME.2019.16.001
```