import torch

from models import DiT_models

if __name__ == "__main__":

    device = "cuda:0"

    model = DiT_models['DiT-S/2'](
        input_size = 32
    ).to(device)

    emb = torch.rand((64, 98304), device=device)
    input = torch.rand((64, 4, 32, 32), device=device)
    t = torch.randint(0, 1000, (input.shape[0],), device=device)

    drop_ids = torch.rand(emb.shape[0], device=device) < 0.1
    emb[drop_ids == 1] = 0

    output = model(input, t, y=emb)