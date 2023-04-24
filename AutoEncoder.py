import torch
import torch.nn as nn
import numpy as np
import wandb
from torch.utils.data import Dataset, DataLoader
from data import load_gmd_hvo_sequences
from logging import getLogger, DEBUG
import yaml
import argparse


logger = getLogger("VAE.py")
logger.setLevel(DEBUG)

parser = argparse.ArgumentParser()

parser.add_argument("--wandb", help="log to wandb", default=True)

# wandb parameters
parser.add_argument(
    "--config",
    help="Yaml file for configuration. If available, the rest of the arguments will be ignored",
    default=None,
)
parser.add_argument("--wandb_project", help="WANDB Project Name", default="AE-Practice")

parser.add_argument("--epochs", help="Number of epochs", default=10)
parser.add_argument("--batch_size", help="Batch size", default=128)
parser.add_argument("--lr", help="Learning rate", default=1e-3)
parser.add_argument("--input_dim", help="Input dim", default=96)
parser.add_argument("--encoder_first_dim", help="Encoder first dim", default=64)
parser.add_argument("--latent_dim", help="Latent dim", default=16)
parser.add_argument("--decoder_output_dim", help="Decoder output dim", default=64)
parser.add_argument("--output_dim", help="Output dim", default=288)

# --------------------------------------------------------------------
# Dummy arguments for running the script in pycharm's python console
# --------------------------------------------------------------------
parser.add_argument("--mode", help="IGNORE THIS PARAM", default="client")
parser.add_argument("--port", help="IGNORE THIS PARAM", default="config.yaml")
# --------------------------------------------------------------------

args, unknown = parser.parse_known_args()
if unknown:
    logger.warning(f"Unknown arguments: {unknown}")

if args.config is not None:
    with open(args.config, "r") as f:
        hparams = yaml.safe_load(f)
else:
    hparams = dict(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        input_dim=args.input_dim,
        encoder_first_dim=args.encoder_first_dim,
        latent_dim=args.latent_dim,
        decoder_output_dim=args.decoder_output_dim,
        output_dim=args.output_dim
    )

# config files without wandb_project specified
if args.wandb_project is not None:
    hparams["wandb_project"] = args.wandb_project

assert "wandb_project" in hparams.keys(), "wandb_project not specified"


# create our data class that implements init, len, getitem
class MonotonicGrooveDataset(Dataset):

    def __init__(self, dataset_setting_json_path, subset_tag, max_len, tapped_voice_idx=2,
                 load_as_tensor=True, collapse_tapped_sequence=False):
        self.flat_seq = None
        self.inputs = list()
        self.outputs = list()
        self.hvo_sequences = list()

        # this sets subset equal to deserialized pickle data which is a list of HVO_Sequence class objects
        subset = load_gmd_hvo_sequences(dataset_setting_json_path, subset_tag, force_regenerate=False)

        for idx, hvo_seq in enumerate(subset):
            if hvo_seq.hits is not None:
                # Adjusts the length of the hvo sequence to the specified number of steps.
                hvo_seq.adjust_length(max_len)
                if np.any(hvo_seq.hits):
                    self.hvo_sequences.append(hvo_seq)
                    flat_seq = hvo_seq.flatten_voices(voice_idx=tapped_voice_idx, reduce_dim=collapse_tapped_sequence)
                    self.flat_seq = flat_seq
                    self.inputs.append(flat_seq)
                    self.outputs.append(hvo_seq.hvo)

        if load_as_tensor:
            self.inputs = torch.tensor(np.array(self.inputs), dtype=torch.float32)
            self.outputs = torch.tensor(np.array(self.outputs), dtype=torch.float32)

    def __len__(self):
        return len(self.hvo_sequences)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx], idx


class Encoder(nn.Module):

    def __init__(self, input_dim, encoder_first_dim, latent_dim, device):
        super().__init__()

        assert np.log2(encoder_first_dim).is_integer(), "First dim should be power of 2"
        assert np.log2(latent_dim).is_integer(), "latent dim should be power of 2"

        if encoder_first_dim == latent_dim:
            n_layers = 1
            scale_dim_by = 1
        else:
            n_layers = int(np.log2(max(encoder_first_dim, latent_dim) / min(encoder_first_dim, latent_dim)))
            scale_dim_by = 2 if latent_dim > encoder_first_dim else 0.5
        self.flatten = nn.Flatten()

        self.layers = []
        _activate_last_layer = False

        self.layers.append(nn.Linear(input_dim, encoder_first_dim).to(self.device))
        self.layers.append(nn.ReLU().to(self.device))

        # encoder
        for i in range(n_layers):
            self.layers.append(nn.Linear(encoder_first_dim, int(encoder_first_dim * scale_dim_by)).to(self.device))
            if i < (n_layers - 1) or _activate_last_layer:
                self.layers.append(nn.ReLU().to(self.device))
            encoder_first_dim = int(encoder_first_dim * scale_dim_by)
        print(self.layers)

    def forward(self, input_data):
        flattened_data = self.flatten(input_data).to(self.device)
        for i, layer in enumerate (self.layers):
            if i == 0:
                x = layer(flattened_data)
            else:
                x = layer(x)
        return x


class Decoder(nn.Module):

    def __init__(self, latent_dim, decoder_output_dim, device):
        super().__init__()

        assert np.log2(decoder_output_dim).is_integer(), "Output dim should be power of 2"
        assert np.log2(latent_dim).is_integer(), "Latent dim should be power of 2"

        if decoder_output_dim == latent_dim:
            n_layers = 1
            scale_dim_by = 1
        else:
            n_layers = int(np.log2(max(latent_dim, decoder_output_dim) / min(latent_dim, decoder_output_dim)))
            scale_dim_by = 2 if decoder_output_dim > latent_dim else 0.5

        self.layers = []

        # decoder
        for i in range(n_layers):
            self.layers.append(nn.Linear(latent_dim, int(latent_dim * scale_dim_by)).to(device))
            latent_dim = int(latent_dim * scale_dim_by)
        print(self.layers)

    def forward(self, encoded_data):
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = layer(encoded_data)
            else:
                x = layer(x)
        return x


class OutputLayer(nn.Module):

    def __init__(self, decoder_output_dim, output_dim, device):
        super().__init__()

        # final output
        self.linear_h = nn.Linear(decoder_output_dim, output_dim).to(device)
        self.linear_v = nn.Linear(decoder_output_dim, output_dim).to(device)
        self.linear_o = nn.Linear(decoder_output_dim, output_dim).to(device)

    def forward(self, decoded_data):
        h = self.linear_h(decoded_data)
        v = self.linear_v(decoded_data)
        o = self.linear_o(decoded_data)
        return h, v, o


# define our nn model
class AutoEncoder(nn.Module):

    def __init__(self, config, device):
        super().__init__()

        # Layers
        # ---------------------------------------------------
        self.Encoder = Encoder(config.input_dim, config.encoder_first_dim, config.latent_dim, device).to(device)
        self.Decoder = Decoder(config.latent_dim, config.decoder_output_dim, device).to(device)
        self.OutputLayer = OutputLayer(config.decoder_output_dim, config.output_dim, device).to(device)

    def forward(self, input_data):
        encoded_data = self.Encoder(input_data)
        decoded_data = self.Decoder(encoded_data)
        h, v, o = self.OutputLayer(decoded_data)
        return h, v, o


def batch_loop(model, data_loader, loss_fn, optimiser, device, no_grad):
    loss_dict = {"h_loss": [],
                 "v_loss": [],
                 "o_loss": [],
                 "total_loss": [],
                 "total_hits_mean": []}

    for inputs, targets, idx in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        h_targets = targets[:, :, :9]
        v_targets = targets[:, :, 9:18]
        o_targets = targets[:, :, 18:]

        # calculate loss
        h_predictions, v_predictions, o_predictions = model(inputs)

        h_predictions = h_predictions.view(h_predictions.shape[0], 32, 9)
        v_predictions = v_predictions.view(h_predictions.shape[0], 32, 9)
        o_predictions = o_predictions.view(h_predictions.shape[0], 32, 9)

        h_loss = loss_fn(h_predictions, h_targets)
        v_loss = loss_fn(v_predictions, v_targets)
        o_loss = loss_fn(o_predictions, o_targets + 0.5)

        h = nn.functional.sigmoid(h_predictions)
        h = torch.where(h > 0.5, 1.0, 0.0)
        # this gives a tensor that has a hit total for each example, so it is 64 length, each value is total hit
        # append all of them into a single list
        total_hits_mean = h.sum(dim=-1).sum(dim=-1).mean()
        # append all values for every batch a plot the mean
        v = nn.functional.sigmoid(v_predictions)
        o = nn.functional.sigmoid(o_predictions)

        total_loss = h_loss + v_loss + o_loss

        loss_dict["h_loss"].append(h_loss)
        loss_dict["v_loss"].append(v_loss)
        loss_dict["o_loss"].append(o_loss)
        loss_dict["total_loss"].append(total_loss)
        loss_dict["total_hits_mean"].append(total_hits_mean)

        # # # backpropagation loss and update weights
        if no_grad is False:
            optimiser.zero_grad()
            total_loss.backward()
            optimiser.step()

    loss_dict["h_loss"] = average(loss_dict["h_loss"])
    loss_dict["v_loss"] = average(loss_dict["v_loss"])
    loss_dict["o_loss"] = average(loss_dict["o_loss"])
    loss_dict["total_loss"] = average(loss_dict["total_loss"])
    loss_dict["total_hits_mean"] = average(loss_dict["total_hits_mean"])

    return loss_dict


# For a "predict" or inference model, sigmoid needs to be applied manually
def train_loop(model, data_loader, loss_fn, optimiser, device):
    loss_dict = batch_loop(model, data_loader, loss_fn, optimiser, device, False)
    loss_dict = {f"train/{key}": value for key, value in loss_dict.items()}
    return loss_dict


# For a "predict" or inference model, sigmoid needs to be applied manually
def test_loop(model, data_loader, loss_fn, optimiser, device):
    loss_dict = batch_loop(model, data_loader, loss_fn, optimiser, device, True)
    loss_dict = {f"test/{key}": value for key, value in loss_dict.items()}
    return loss_dict


def average(lst):
    return sum(lst) / len(lst)


if __name__ == "__main__":

    # Initialize wandb
    # ----------------------------------------------------------------------------------------------------------
    wandb_run = wandb.init(
        config=hparams,  # either from config file or CLI specified hyperparameters
        project=hparams["wandb_project"],  # name of the project
        anonymous="allow",
        settings=wandb.Settings(code_dir="VAE.py")  # for code saving
    )

    # Reset config to wandb.config (in case of sweeping with YAML necessary)
    # ----------------------------------------------------------------------------------------------------------
    config = wandb.config
    run_name = wandb_run.name
    run_id = wandb_run.id

    # Load Training and Testing Datasets and Wrap them in torch.utils.data.Dataloader
    # ----------------------------------------------------------------------------------------------------------
    training_data = MonotonicGrooveDataset(
        dataset_setting_json_path="data/dataset_json_settings/4_4_Beats_gmd.json",
        subset_tag="train",
        max_len=32,
        tapped_voice_idx=2,
        load_as_tensor=True,
        collapse_tapped_sequence=True)

    testing_data = MonotonicGrooveDataset(
        dataset_setting_json_path="data/dataset_json_settings/4_4_Beats_gmd.json",
        subset_tag="test",
        max_len=32,
        tapped_voice_idx=2,
        load_as_tensor=True,
        collapse_tapped_sequence=True)

    # initialize the data loader with our training data
    train_data_loader = DataLoader(training_data, batch_size=config.batch_size, shuffle=True)

    # initialize the data loader with our test data
    test_data_loader = DataLoader(testing_data, batch_size=config.batch_size, shuffle=True)

    # initialize model
    gpu1 = torch.device("cuda")
    device = gpu1 if torch.cuda.is_available() else "cpu"
    auto_encoder = AutoEncoder(config, device).to(device)

    # instantiate loss fn and optimiser
    bce_with_logits_loss_fn = nn.BCEWithLogitsLoss()
    adam_optimiser = torch.optim.Adam(auto_encoder.parameters(), lr=config.lr)

    for i in range(config.epochs):
        print(f"Epoch {i+1}")
        train_loss_dict = train_loop(auto_encoder, train_data_loader, bce_with_logits_loss_fn, adam_optimiser, gpu1 if torch.cuda.is_available() else "cpu")
        wandb.log(train_loss_dict, commit=False)
        test_loss_dict = test_loop(auto_encoder, test_data_loader, bce_with_logits_loss_fn, adam_optimiser, gpu1 if torch.cuda.is_available() else "cpu")
        wandb.log(test_loss_dict, commit=False)
        wandb.log({"epoch": i}, commit=True)
        print("-------------------")
    print("testing done")

    wandb.finish()


