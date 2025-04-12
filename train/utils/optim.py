import torch

def get_optimizer(name, model, lr):
    if name == "Adam" or name.lower() == "adam":
        print("Optimizer Adam")
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif name == "SGD" or name.lower() == "sgd":
        print("Optimizer SGD")
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif name == "Adadelta" or name.lower() == "adadelta":
        print("Optimizer Adadelta")
        return torch.optim.Adadelta(model.parameters(), lr=lr)
    elif name == "AdamW" or name.lower() == "adamw":
        print("Optimizer AdamW")
        return torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        print("Optimizer {} not found".format(name))
        print("Auto select Adam")
        return torch.optim.Adam(model.parameters(), lr=lr)


def get_scheduler(name, optimizer, step_size, gamma=0.1):
    if name == "StepLR" or name.lower() == "steplr":
        print("Scheduler StepLR")
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    elif name == "ExponentialLR" or name.lower() == "exponentiallr":
        print("Scheduler ExponentialLR")
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=gamma
        )
    elif name == "ReduceLROnPlateau" or name.lower() == "reducelronplateau":
        print("Scheduler ReduceLROnPlateau")
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=gamma, threshold=0.02
        )
    elif name == "cosineannealinglr" or name.lower() == "cosineannealinglr":
        print("Scheduler CosineAnnealingLR")
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(step_size * 3)
        )
    else:
        print("Scheduler {} not found".format(name))
        print("Auto select StepLR")
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
