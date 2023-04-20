import os
import yaml
import argparse

import torch
import torch.optim as optim
import torchsummary
import torchvision.transforms as TVF
import torch.nn.functional as TNF
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from backstage.dataset import DogDataset
from backstage.models import Classifier


# TODO use better function from library
def getAccuracy(yhat: float, y: float) -> float:
    """Get accuracy of prediction."""
    B = y.size(0)
    yhatmax = torch.argmax(yhat, 1)
    ymax = torch.argmax(y, 1)
    r = (yhatmax == ymax) * 1.0
    acc = r.sum() / B
    return acc


def saveCheckpoint(model: Classifier, opt: optim, fn: str) -> None:
    """Save checkpoint state of model at given point.
    
        Args
            model: model to save checkpoint from
            opt: used optimizer
            fn: file to save checkpoint at
    """
    checkpoint = {
        "model": model.state_dict(),
        "opt": opt.state_dict()
    }
    torch.save(checkpoint, fn)


class Avg:
    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def reset(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def step(self, value):
        self.count += 1
        self.sum += value
        self.avg = self.sum / self.count


def loadDogDataset(config: dict, fileName: str) -> DataLoader:
    """
        Load dataset containing dog images and transform them.

        Args:
            config: configuration about the file
            fileName: name of file to load

            Returns:
            Dataloader object with loaded info about images
    """

    inputTransform = TVF.Compose(
        [
            TVF.Resize(size=(config["imageSize"][0], config["imageSize"][1])),
            TVF.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    ds = DogDataset(
        folder=config["dataFolder"], 
        h5_filename=fileName,
        scale_shape=None,
        as_tensor=True,
        transform=inputTransform,
        detectionGridSize=config["detectionGridSize"],
        imageSize=config["imageSize"],
        localization=False,
        apply_random_sized_bbox_safe_crop=False,
    )

    return DataLoader(
        ds, batch_size=config["batchSize"],
        num_workers=config["numWorkers"],
        shuffle=True
        )


def stageTrain(config: yaml):
    """
        Load dataset 
    """
    wandb.init(
        project="cnn_exercise",
        entity="loptosi-team",
        config=config,
        group=config.get("group"),
        mode=config.get('wandb_mode', 'online')
    )
    device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    loaderTrain = loadDogDataset(config, config["dataTrain"])
    loaderVal   = loadDogDataset(config, config["dataVal"])

    # Create model
    model = Classifier(args=config["models"], nCategories=config["nCategories"])
    print(model)
    model = model.to(device)
    model.train()

    torchsummary.summary(model, input_size=(3, config["imageSize"][0], config["imageSize"][1]))

    # Optimizer
    opt = optim.Adam(params=model.get_params(), lr=config["lr"], weight_decay=config["optim_decay"])

    criterion_presence = torch.nn.BCEWithLogitsLoss()
    criterion_coords = torch.nn.MSELoss()

    # Create checkpoint files if user wants them
    if not os.path.exists("data/checkpoints/"):
        os.makedirs("data/checkpoints/")

    if config["saveCheckpoints"]:
        fn = "data/checkpoints/epoch-0.pt"
        if (os.path.isfile(fn)):
            checkpoint = torch.load(fn, map_location=torch.device("cpu"))
            model.load_state_dict(checkpoint["model"])
            opt.load_state_dict(checkpoint["opt"])
            model = model.to(device)

    wandb.init(project="cnn_exercise_localization", entity="loptosi-team", config=config)

    for epoch in range(config["epochs"]):

        avg_train_presence = Avg()
        avg_train_coords = Avg()

        print(f"Training epoch: {epoch}")

        progress = tqdm(loaderTrain, ascii=True)
        model.train()
        for (x, y) in progress:
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)

            train_presence = y[:, 0:1]
            train_coord = y[:, 1:5]
            train_presence_hat = y_hat[:, 0:1]
            train_valid_coord_hat = y_hat[:, 1:5]

            # Optimalisation
            opt.zero_grad()
            loss_train_presence = criterion_presence(train_presence_hat, train_presence)
            loss_train_coords = criterion_coords(
                train_valid_coord_hat * train_presence, train_coord * train_presence
            )
            loss = loss_train_presence + loss_train_coords

            loss.backward()
            opt.step()

            # extract NMS and calculate precision

            # Compute stats
            avg_train_coords.step(loss_train_coords.cpu().detach().item())
            avg_train_presence.step(loss_train_presence.cpu().detach().item())

            info = {"coords_loss": avg_train_coords.avg, "presence_Loss": avg_train_presence.avg}

            progress.set_postfix(info)

        if config["saveCheckpoints"]:
            fn = "data/checkpoints/epoch-{}.pt".format(epoch)
            print("Saving epoch: ", fn)
            saveCheckpoint(model, opt, fn)

        print(f"Validating epoch: {epoch}")

        progress = tqdm(loaderVal, ascii=True)
        model.eval()

        avg_valid_presence = Avg()
        avg_valid_coords = Avg()
        with torch.no_grad():
            for (x, y) in progress:

                x = x.to(device)
                y = y.to(device)
                y_hat = model(x)

                valid_presence = y[:, 0:1]
                valid_coord = y[:, 1:5]
                valid_presence_hat = y_hat[:, 0:1]
                valid_coord_hat = y_hat[:, 1:5]

                loss_valid_presence = criterion_presence(valid_presence_hat, valid_presence)
                loss_valid_coords = criterion_coords(
                    valid_coord_hat * valid_presence, valid_coord * valid_presence
                )
                loss = loss_valid_presence + loss_valid_coords

                avg_valid_coords.step(loss_valid_coords.cpu().detach().item())
                avg_valid_presence.step(loss_valid_presence.cpu().detach().item())

                info = {"coords_loss": avg_valid_coords.avg, "presence_loss": avg_valid_presence.avg}

                progress.set_postfix(info)

        log = {
            "train_loss": avg_train_presence.avg + avg_train_coords.avg,
            "valid_loss": avg_valid_presence.avg + avg_valid_coords.avg,
            "train_presence_avg_loss": avg_train_presence.avg,
            "train_coords_avg_loss": avg_train_coords.avg,
            "valid_presence_avg_loss": avg_valid_presence.avg,
            "valid_coords_avg_loss": avg_valid_coords.avg,
        }
        wandb.log(log)

    wandb.finish()


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", default='online')
    args = parser.parse_args()

    file = dir_path + "/../params.yaml"
    with open(file, mode="r") as f:
        config = yaml.safe_load(f)

    param_cfg = config['train']
    param_cfg['wandb_mode'] = args.wandb
    stageTrain(param_cfg)
