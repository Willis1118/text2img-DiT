import torch 
import torch_xla.core.xla_model as xm

if __name__ == "__main__":
    print(xm.xla_device())