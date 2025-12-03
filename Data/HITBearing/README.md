#### 1. working condition:
| Code  | Working condition (LP_HP) |
| ----- | ------------------------- |
| load0 | 1000_1200 r/min           |
| load1 | 1500_1800 r/min           |
| load2 | 2000_2400 r/min           |
| load3 | 2500_3000 r/min           |

#### 2. Lable:
| Label | Bearing code       |
| ----- | ------------------ |
| 0     | normal             |
| 1     | inner race 0.5_0.5 |
| 2     | inner race 0.5_1.0 |
| 3     | outer race 0.5_0.5 |

#### 3. Data structure of each file:
```
{
    label 0: numpy.ndarray # [number of sequence, sequence length]
    label 1: numpy.ndarray
    label 2: numpy.ndarray
    label 3: numpy.ndarray
}
```

#### 4. How to load data
```python
>>> import pickle
>>> with open('load0.pkl', 'rb') as ff:
        data = pickle.load(ff)
>>> print(data[0].shape)
(360, 1024)
```
This means: in load 0, label 0 contains 360 sequences with a length of 1024.

**NOTE**: All data have been normalized between 0 and 1.

#### 5. Citation

```
Hou, L., H. Yi, Y. Jin, M. Gui, L. Sui, J. Zhang, and Y. Chen. “Inter-Shaft Bearing Fault Diagnosis Based on Aero-Engine System: A Benchmarking Dataset Study”. Journal of Dynamics, Monitoring and Diagnostics, vol. 2, no. 4, Aug. 2023, pp. 228-42, doi:10.37965/jdmd.2023.314.
```