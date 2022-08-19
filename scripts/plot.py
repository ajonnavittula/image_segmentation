import matplotlib.pyplot as plt
import numpy as np


data = [1.782, 33.518, 57.432, 62.994,  72, 43.27]
names = ["FAT", "DoPose", "SPS", "DoPose + SPS", "FAT+Dopose+SPS", "IAS"]

plt.bar(np.arange(len(data)), data)
plt.xticks(np.arange(len(data)), names)
plt.xlabel("Models")
plt.ylabel("Average Precison (AP)")
plt.tight_layout()
plt.show()