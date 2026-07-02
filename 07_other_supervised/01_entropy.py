import numpy as np
import matplotlib.pyplot as plt

p = np.arange(0.01, 1, 0.01)
entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
plt.plot(p, entropy)
plt.xlabel('p')
plt.ylabel('Entropy')
plt.title('Entropy vs p')
plt.show()
