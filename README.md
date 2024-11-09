# Description
This is a collection of python tools used in [my work](https://arxiv.org/abs/2411.03988)

# Usage
You simply add the packages into your PATH

For instance, I put the whole file folder in `/home/dyliu/`, 
so
```py
import sys
sys.path.append('/home/dyliu/mytools/bin/')
```

Then you are free to use, for example:
```py
from calculation import freq2z, u
freq2z(1.3*u.GHz)
```

# Dependency
* [Python](https://www.python.org/) version: 3.9
* packages:
    * [numpy](https://numpy.org/)
    * [matplotlib](https://matplotlib.org/)
    * [h5py](https://www.h5py.org/)
    * [astropy](https://www.astropy.org/)
    * [scipy](https://scipy.org/)
