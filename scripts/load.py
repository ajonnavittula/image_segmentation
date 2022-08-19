import torch

loaded = torch.jit.load('../export2/model.ts')

print(loaded)
print(loaded.code)