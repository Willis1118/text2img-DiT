import wandb

def initialize_wandb(args, exp_name):
    wandb.login()

    wandb.init(
        project="t2i-DiT",
        name=exp_name
    )
    wandb.config.update(args)

def log_loss_dict(loss_dict, steps):
    wandb.log({k: v for k, v in loss_dict.items()}, step=steps)

def log_images(samples, prefix, steps):
    wandb.log({f"{prefix}-samples": wandb.Image(samples)}, step=steps)

if __name__ == "__main__":
    wandb_configs = {
        "model": "test_model",
        "epochs": 1,
        "learning_rate": 1e-4,
        "batch_size": 128,
    }

    initialize_wandb(wandb_configs, exp_name="test_exp")