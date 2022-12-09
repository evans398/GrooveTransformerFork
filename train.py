import os

import wandb
import torch
from model import GrooveTransformerEncoderVAE
from helpers import vae_train_utils, vae_test_utils
from data.src.dataLoaders import MonotonicGrooveDataset
from torch.utils.data import DataLoader
from logging import getLogger, DEBUG
import yaml
import argparse
import numpy as np

logger = getLogger("train.py")
logger.setLevel(DEBUG)

# logger.info("MAKE SURE YOU DO THIS")
# logger.warning("this is a warning!")

parser = argparse.ArgumentParser()

# ----------------------- Set True When Testing ----------------
parser.add_argument("--is_testing", help="Use testing dataset (1% of full date) for testing the script", type=bool,
                    default=False)

# ----------------------- WANDB Settings -----------------------
parser.add_argument("--wandb", type=bool, help="log to wandb", default=True)
# wandb parameters
parser.add_argument(
    "--config",
    help="Yaml file for configuration. If available, the rest of the arguments will be ignored", default=None)
parser.add_argument("--wandb_project", type=str, help="WANDB Project Name", default="voice_distribution_and_genre_distribution_imbalance")

# ----------------------- Model Parameters -----------------------
# d_model_dec_ratio denotes the ratio of the dec relative to enc size
parser.add_argument("--d_model_enc", type=int, help="Dimension of the encoder model", default=32)
parser.add_argument("--d_model_dec_ratio", type=int,help="Dimension of the decoder model as a ratio of d_model_enc", default=1)
parser.add_argument("--embedding_size_src", type=int, help="Dimension of the source embedding", default=3)
parser.add_argument("--embedding_size_tgt",  type=int, help="Dimension of the target embedding", default=27)
parser.add_argument("--nhead_enc", type=int, help="Number of attention heads for the encoder", default=2)
parser.add_argument("--nhead_dec", type=int, help="Number of attention heads for the decoder", default=2)
# d_ff_enc_to_dmodel denotes the ratio of the feed_forward ratio in encoder relative to the encoder dim (d_model_enc)
parser.add_argument("--d_ff_enc_to_dmodel", type=float, help="ration of the dimension of enc feed-frwrd layer relative to "
                                                 "enc dmodel", default=1)
# d_ff_dec_to_dmodel denotes the ratio of the feed_forward ratio in encoder relative to the encoder dim (d_model_enc)
parser.add_argument("--d_ff_dec_to_dmodel", type=float,
                    help="ration of the dimension of dec feed-frwrd layer relative to decoder dmodel", default=1)
# n_dec_lyrs_ratio denotes the ratio of the dec relative to n_enc_lyrs
parser.add_argument("--n_enc_lyrs", type=int, help="Number of encoder layers", default=3)
parser.add_argument("--n_dec_lyrs_ratio", type=float, help="Number of decoder layers as a ratio of "
                                               "n_enc_lyrs as a ratio of d_ff_enc", default=1)
parser.add_argument("--max_len_enc", type=int, help="Maximum length of the encoder", default=32)
parser.add_argument("--max_len_dec", type=int, help="Maximum length of the decoder", default=32)
parser.add_argument("--latent_dim", type=int, help="Overall Dimension of the latent space", default=16)

# ----------------------- Loss Parameters -----------------------
parser.add_argument("--hit_loss_balancing_beta", type=float, help="beta parameter for hit loss balancing", default=0.0)
parser.add_argument("--genre_loss_balancing_beta", type=float, help="beta parameter for genre loss balancing", default=0.0)
parser.add_argument("--hit_loss_function", type=str, help="hit_loss_function - only bce supported for now", default="bce")
parser.add_argument("--velocity_loss_function", type=str, help="velocity_loss_function - either 'bce' or 'mse' loss",
                    default='bce', choices=['bce', 'mse'])
parser.add_argument("--offset_loss_function", type=str, help="offset_loss_function - either 'bce' or 'mse' loss",
                    default='bce', choices=['bce', 'mse'])

# ----------------------- Training Parameters -----------------------
parser.add_argument("--dropout", type=float, help="Dropout", default=0.4)
parser.add_argument("--force_data_on_cuda", type=bool, help="places all training data on cude", default=True)
parser.add_argument("--epochs", type=int, help="Number of epochs", default=100)
parser.add_argument("--batch_size", type=int, help="Batch size", default=64)
parser.add_argument("--lr", type=float, help="Learning rate", default=1e-4)
parser.add_argument("--optimizer", type=str, help="optimizer to use - either 'sgd' or 'adam' loss", default="sgd",
                    choices=['sgd', 'adam'])

# ----------------------- Data Parameters -----------------------
parser.add_argument("--dataset_json_dir", type=str,
                    help="Path to the folder hosting the dataset json file",
                    default="data/dataset_json_settings")
parser.add_argument("--dataset_json_fname", type=str,
                    help="filename of the data (USE 4_4_Beats_gmd.jsom for only beat sections,"
                         " and 4_4_BeatsAndFills_gmd.json for beats and fill samples combined",
                    default="4_4_Beats_gmd.json")
parser.add_argument("--evaluate_on_subset", type=str,
                    help="Using test or evaluation subset for evaluating the model", default="test",
                    choices=['test', 'evaluation'] )

# ----------------------- Evaluation Params -----------------------
parser.add_argument("--calculate_hit_scores_on_train", type=bool,
                    help="Evaluates the quality of the hit models on training set",
                    default=True)
parser.add_argument("--calculate_hit_scores_on_test", type=bool,
                    help="Evaluates the quality of the hit models on test/evaluation set",
                    default=True)
parser.add_argument("--piano_roll_samples", type=bool, help="Generate audio samples", default=True)
parser.add_argument("--piano_roll_frequency", type=int, help="Frequency of piano roll generation", default=20)
parser.add_argument("--hit_score_frequency", type=int, help="Frequency of hit score generation", default=10)

# ----------------------- Misc Params -----------------------
parser.add_argument("--save_model", type=bool, help="Save model", default=True)
parser.add_argument("--save_model_dir", type=str, help="Path to save the model", default="misc/VAE")
parser.add_argument("--save_model_frequency", type=int, help="Save model every n epochs", default=100)


args, unknown = parser.parse_known_args()
if unknown:
    logger.warning(f"Unknown arguments: {unknown}")

# Disable wandb logging in testing mode
if args.is_testing:
    os.environ["WANDB_MODE"] = "disabled"

if args.config is not None:
    with open(args.config, "r") as f:
        hparams = yaml.safe_load(f)
else:
    d_model_dec = int(float(args.d_model_enc) * float(args.d_model_dec_ratio))
    dim_feedforward_enc = int(float(args.d_ff_enc_to_dmodel)*float(args.d_model_enc))
    dim_feedforward_dec = int(float(args.d_ff_dec_to_dmodel) * d_model_dec)
    num_decoder_layers = int(float(args.n_enc_lyrs) * float(args.n_dec_lyrs_ratio))
    hparams = dict(
        d_model_enc=args.d_model_enc,
        d_model_dec=d_model_dec,
        dim_feedforward_enc=dim_feedforward_enc,
        dim_feedforward_dec=dim_feedforward_dec,
        num_encoder_layers=int(args.n_enc_lyrs),
        num_decoder_layers=num_decoder_layers,
        embedding_size_src=args.embedding_size_src,
        embedding_size_tgt=args.embedding_size_tgt,
        nhead_enc=args.nhead_enc,
        nhead_dec=args.nhead_dec,
        dropout=args.dropout,
        latent_dim=args.latent_dim,
        max_len_enc=args.max_len_enc,
        max_len_dec=args.max_len_dec,
        o_activation="tanh" if args.offset_loss_function=="mse" else "sigmoid",
        hit_loss_function=args.hit_loss_function,
        velocity_loss_function=args.velocity_loss_function,
        offset_loss_function=args.offset_loss_function,
        hit_loss_balancing_beta=float(args.hit_loss_balancing_beta),
        genre_loss_balancing_beta=float(args.genre_loss_balancing_beta),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        optimizer=args.optimizer,
        is_testing=args.is_testing,
        dataset_json_dir=args.dataset_json_dir,
        dataset_json_fname=args.dataset_json_fname,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )


# config files without wandb_project specified
if args.wandb_project is not None:
    hparams["wandb_project"] = args.wandb_project

assert "wandb_project" in hparams.keys(), "wandb_project not specified"


if __name__ == "__main__":

    # Initialize wandb
    # ----------------------------------------------------------------------------------------------------------
    wandb_run = wandb.init(
        config=hparams,                         # either from config file or CLI specified hyperparameters
        project=hparams["wandb_project"],          # name of the project
        anonymous="allow",
        entity="mmil_vae_g2d",                          # saves in the mmil_vae_g2d team account
        settings=wandb.Settings(code_dir="train.py")    # for code saving
    )

    # Reset config to wandb.config (in case of sweeping with YAML necessary)
    # ----------------------------------------------------------------------------------------------------------
    config = wandb.config
    run_name = wandb_run.name
    run_id = wandb_run.id
    collapse_tapped_sequence = (args.embedding_size_src == 3)
    # Load Training and Testing Datasets and Wrap them in torch.utils.data.Dataloader
    # ----------------------------------------------------------------------------------------------------------
    # only 1% of the dataset is used if we are testing the script (is_testing==True)
    should_place_all_data_on_cuda = args.force_data_on_cuda and torch.cuda.is_available()
    training_dataset = MonotonicGrooveDataset(
        dataset_setting_json_path="data/dataset_json_settings/4_4_Beats_gmd.json",
        subset_tag="train",
        max_len=int(args.max_len_enc),
        tapped_voice_idx=2,
        collapse_tapped_sequence=collapse_tapped_sequence,
        down_sampled_ratio=0.1 if args.is_testing is True else None,
        move_all_to_gpu=should_place_all_data_on_cuda,
        hit_loss_balancing_beta=args.hit_loss_balancing_beta,
        genre_loss_balancing_beta=args.genre_loss_balancing_beta
    )
    train_dataloader = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True)

    test_dataset = MonotonicGrooveDataset(
        dataset_setting_json_path="data/dataset_json_settings/4_4_Beats_gmd.json",
        subset_tag="test",
        max_len=int(args.max_len_enc),
        tapped_voice_idx=2,
        collapse_tapped_sequence=collapse_tapped_sequence,
        down_sampled_ratio=0.1 if args.is_testing is True else None,
        move_all_to_gpu=should_place_all_data_on_cuda
    )

    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    # Initialize the model
    # ------------------------------------------------------------------------------------------------------------
    groove_transformer_vae_cpu = GrooveTransformerEncoderVAE(config)

    groove_transformer_vae = groove_transformer_vae_cpu.to(config.device)
    wandb.watch(groove_transformer_vae, log="all", log_freq=1)

    # Instantiate the loss Criterion and Optimizer
    # ------------------------------------------------------------------------------------------------------------

    if config.hit_loss_function == "bce":
        hit_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    else:
        raise NotImplementedError(f"hit_loss_function {config.hit_loss_function} not implemented")

    if config.velocity_loss_function == "bce":
        velocity_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    else:
        velocity_loss_fn = torch.nn.MSELoss(reduction='none')

    if config.offset_loss_function == "bce":
        offset_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    else:
        offset_loss_fn = torch.nn.MSELoss(reduction='none')

    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(groove_transformer_vae.parameters(), lr=config.lr)
    else:
        optimizer = torch.optim.SGD(groove_transformer_vae.parameters(), lr=config.lr)

    # Iterate over epochs
    # ------------------------------------------------------------------------------------------------------------
    metrics = dict()
    step_ = 0


    def frange_cycle_sigmoid(start, stop, n_epoch, n_cycle=4, ratio=0.5):
        L = np.ones(n_epoch)
        period = n_epoch / n_cycle
        step = (stop - start) / (period * ratio)  # step is in [0,1]

        # transform into [-6, 6] for plots: v*12.-6.

        for c in range(n_cycle):

            v, i = start, 0
            while v <= stop:
                L[int(i + c * period)] = 1.0 / (1.0 + np.exp(- (v * 12. - 6.)))
                v += step
                i += 1
        return L


    beta_np_cyc = frange_cycle_sigmoid(start=0.0, stop=1, n_epoch=config.epochs, n_cycle=1, ratio=0.25)

    for epoch in range(config.epochs):
        print(f"Epoch {epoch} of {config.epochs}, steps so far {step_}")

        # Run the training loop (trains per batch internally)
        # ------------------------------------------------------------------------------------------
        groove_transformer_vae.train()

        logger.info("***************************Training...")

        train_log_metrics, step_ = vae_train_utils.train_loop(
            train_dataloader=train_dataloader,
            groove_transformer_vae=groove_transformer_vae,
            optimizer=optimizer,
            hit_loss_fn=hit_loss_fn,
            velocity_loss_fn=velocity_loss_fn,
            offset_loss_fn=offset_loss_fn,
            device=config.device,
            starting_step=step_,
            kl_beta=beta_np_cyc[epoch])

        wandb.log(train_log_metrics, commit=False)
        wandb.log({"kl_beta": beta_np_cyc[epoch]}, commit=False)

        # ---------------------------------------------------------------------------------------------------
        # After each epoch, evaluate the model on the test set
        #     - To ensure not overloading the GPU, we evaluate the model on the test set also in batche
        #           rather than all at once
        # ---------------------------------------------------------------------------------------------------
        groove_transformer_vae.eval()       # DON'T FORGET TO SET THE MODEL TO EVAL MODE (check torch no grad)

        logger.info("***************************Testing...")

        test_log_metrics = vae_train_utils.test_loop(
            test_dataloader=test_dataloader,
            groove_transformer_vae=groove_transformer_vae,
            hit_loss_fn=hit_loss_fn,
            velocity_loss_fn=velocity_loss_fn,
            offset_loss_fn=offset_loss_fn,
            device=config.device,
            kl_beta=beta_np_cyc[epoch])

        wandb.log(test_log_metrics, commit=False)
        logger.info(f"Epoch {epoch} Finished with total train loss of {train_log_metrics['train/loss_total']} "
                    f"and test loss of {test_log_metrics['test/loss_total']}")

        # Generate PianoRolls and KL/OA PLots if Needed
        # ---------------------------------------------------------------------------------------------------
        if args.piano_roll_samples:
            if epoch % args.piano_roll_frequency == 0:
                media = vae_test_utils.get_logging_media_for_vae_model_wandb(
                    groove_transformer_vae=groove_transformer_vae,
                    device=config.device,
                    dataset_setting_json_path=f"{config.dataset_json_dir}/{config.dataset_json_fname}",
                    subset_name='test',
                    down_sampled_ratio=0.005,
                    collapse_tapped_sequence=collapse_tapped_sequence,
                    cached_folder="eval/GrooveEvaluator/templates",
                    divide_by_genre=True,
                    need_piano_roll=True,
                    need_kl_plot=False,
                    need_audio=False
                )
                wandb.log(media, commit=False)

        # Get Hit Scores for the entire train and the entire test set
        # ---------------------------------------------------------------------------------------------------
        if args.calculate_hit_scores_on_train:
            if epoch % args.hit_score_frequency == 0:
                logger.info("________Calculating Hit Scores on Train Set...")
                train_set_hit_scores = vae_test_utils.get_hit_scores_for_vae_model(
                    groove_transformer_vae=groove_transformer_vae,
                    device=config.device,
                    dataset_setting_json_path=f"{config.dataset_json_dir}/{config.dataset_json_fname}",
                    subset_name='train',
                    down_sampled_ratio=0.1,
                    collapse_tapped_sequence=collapse_tapped_sequence,
                    cached_folder="eval/GrooveEvaluator/templates",
                    divide_by_genre=False
                )
                wandb.log(train_set_hit_scores, commit=False)

        if args.calculate_hit_scores_on_test:
            if epoch % args.hit_score_frequency == 0:
                logger.info("________Calculating Hit Scores on Test Set...")
                test_set_hit_scores = vae_test_utils.get_hit_scores_for_vae_model(
                    groove_transformer_vae=groove_transformer_vae,
                    device=config.device,
                    dataset_setting_json_path=f"{config.dataset_json_dir}/{config.dataset_json_fname}",
                    subset_name=args.evaluate_on_subset,
                    down_sampled_ratio=None,
                    collapse_tapped_sequence=collapse_tapped_sequence,
                    cached_folder="eval/GrooveEvaluator/templates",
                    divide_by_genre=False
                )
                wandb.log(test_set_hit_scores, commit=False)

        # Commit the metrics to wandb
        # ---------------------------------------------------------------------------------------------------
        wandb.log({"epoch": epoch}, step=epoch)

        # Save the model if needed
        # ---------------------------------------------------------------------------------------------------
        if args.save_model:
            if epoch % args.save_model_frequency == 0 and epoch > 0:
                if epoch < 10:
                    ep_ = f"00{epoch}"
                elif epoch < 100:
                    ep_ = f"0{epoch}"
                else:
                    ep_ = epoch
                model_artifact = wandb.Artifact(f'model_epoch_{ep_}', type='model')
                model_path = f"{args.save_model_dir}/{args.wandb_project}/{run_name}_{run_id}/{ep_}.pth"
                groove_transformer_vae.save(model_path)
                model_artifact.add_file(model_path)
                wandb_run.log_artifact(model_artifact)
                logger.info(f"Model saved to {model_path}")

    wandb.finish()

