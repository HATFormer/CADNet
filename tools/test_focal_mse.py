import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10,10,1000)

y = (0.2*x**2-1)**2
y1 = x**2
plt.plot(x,y,'r',label='focal_L2_loss')
plt.plot(x,y1,label='l2_loss')

plt.xlabel('x')
plt.ylabel('loss')
plt.legend()
plt.show()