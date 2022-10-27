import torch
from model.src.BasicGrooveTransformer_VAE import *
from helpers.BasicGrooveTransformer_train_VAE import *
# Load dataset as torch.utils.data.Dataset
from data.dataLoaders import MonotonicGrooveDataset
from torch.utils.data import DataLoader



# load dataset as torch.utils.data.Dataset
training_dataset = MonotonicGrooveDataset(
    dataset_setting_json_path="data/dataset_json_settings/4_4_Beats_gmd.json",
    subset_tag="train",
    max_len=32,
    tapped_voice_idx=2,
    collapse_tapped_sequence=False)


### encoder params
nhead_enc = 3
nhead_dec = 3
d_model_enc = 27
d_model_dec = 30
embedding_size_src = 27
embedding_size_tgt = 27
dim_feedforward = 2048
dropout = 0.4
num_encoder_layers = 2
num_decoder_layers = 2
max_len = 32
device = 0

latent_dim = int((max_len * d_model_enc)/4)

### call VAE encoder

groove_transformer = GrooveTransformerEncoderVAE(d_model_enc, d_model_dec, embedding_size_src, embedding_size_tgt,
                 nhead_enc, nhead_dec, dim_feedforward, dropout, num_encoder_layers, latent_dim,
                 num_decoder_layers, max_len, device)

optimizer = torch.optim.Adam(groove_transformer.parameters(), lr=1e-4)
#inputs = torch.rand(20, max_len, embedding_size_src)
# PYTORCH LOSS FUNCTIONS
# BCE used for hit loss
bce_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
# MSE used for velocities and offsets losses
mse_fn = torch.nn.MSELoss(reduction='none')
hit_loss_penalty = 0.1

# run one epoch
groove_transformer.train()  # train mode
optimizer.zero_grad()
# forward + backward + optimize
#output_net = groove_transformer(inputs)

# loss, training_accuracy, training_perplexity, bce_h, mse_v, mse_o = calculate_loss_VAE(output_net, inputs, bce_fn,
#                                                                                        mse_fn,
#                                                                                        hit_loss_penalty)
#
# loss.backward()
# optimizer.step()



### epochs for
LOSS = []
epochs = 1 #10

for epoch in range(epochs):
    # in each epoch we iterate over the entire dataset
    for batch_count, (inputs, outputs, indices) in enumerate(train_dataloader):
        print(f"Epoch {epoch} - Batch #{batch_count} - inputs.shape {inputs.shape} - "
              f"outputs.shape {outputs.shape} - indices.shape {indices.shape} ")

        inputs = inputs.float()
        # run one epoch
        groove_transformer.train()  # train mode
        optimizer.zero_grad()
        # forward + backward + optimize
        output_net = groove_transformer(inputs)
        # loss = calculate_loss_VAE(outputs, labels)

        loss, training_accuracy, training_perplexity, bce_h, mse_v, mse_o = calculate_loss_VAE(output_net, inputs, bce_fn,
                                                                                               mse_fn,
                                                                                               hit_loss_penalty)

        loss.backward()
        optimizer.step()

        LOSS.append(loss)

## plot loss
import matplotlib.pyplot as plt
plt.plot(LOSS)

