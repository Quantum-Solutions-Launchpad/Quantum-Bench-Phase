from utils import haldane_iqpe
from qiskit_nature.second_q.mappers import JordanWignerMapper
import matplotlib.pyplot as plt
import os

# params here

# call haldane iqpe function here
data = {}

print(data)
plt.plot(*zip(*sorted(data.items())))
plt.title("Hubbard Graphene Hexagon BdG Energy vs. Occupation Number")

file_path = os.path.join(os.getcwd(), "..", "plots/haldane-model/hubbard-iqpe.png")
plt.savefig(file_path)