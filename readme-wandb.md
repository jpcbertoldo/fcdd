src: quickstart from their web app

```
pip install wandb
wandb login
```

example code

```Python
import wandb

# At the top of your training script, start a new run
wandb.init(project="test-project", entity="mines-paristech-cmm")

# Capture a dictionary of hyperparameters with config
wandb.config = {
  "learning_rate": 0.001,
  "epochs": 100,
  "batch_size": 128
}

# Log metrics inside your training loop to visualize model performance
wandb.log({"loss": loss})

# Optional
wandb.watch(model)
```