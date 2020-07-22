import seaborn as sns
import numpy as np
from numpy.random import randn
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

f = open('test_000.txt','r')
ls = f.readlines()
data = []
for i in ls:
	i = float(i.strip())
	data.append(i)

plt.hist(data,bins=50)
plt.show()