import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
from sklearn import metrics
import sys 

sys.path.append('/home/skrhakv/cryptic-nn/src')
import baseline_utils
import finetuning_utils

torch.manual_seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from transformers import AutoTokenizer
import torch
import torch.nn as nn
import numpy as np
import functools
from sklearn import metrics
import gc
import bitsandbytes as bnb
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings('ignore')
torch.manual_seed(42)

MODEL_NAME = 'facebook/esm2_t33_650M_UR50D'
DATA_PATH = '/home/skrhakv/nn-for-kamila/data/filtered-LIGYSIS'

finetuned_model = finetuning_utils.FinetunedEsmModel(MODEL_NAME).half().to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = finetuning_utils.process_sequence_dataset(f'{DATA_PATH}/full-train.txt', tokenizer)
val_dataset = finetuning_utils.process_sequence_dataset(f'{DATA_PATH}/validation.txt', tokenizer)

partial_collate_fn = functools.partial(finetuning_utils.collate_fn, tokenizer=tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=partial_collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=val_dataset.num_rows, collate_fn=partial_collate_fn)

optimizer = bnb.optim.AdamW8bit(finetuned_model.parameters(), lr=0.0001, eps=1e-4) 

EPOCHS = 3

# compute class weights
for batch in train_dataloader:
    labels = batch['labels']
class_labels = labels.cpu().numpy().reshape(-1)[labels.cpu().numpy().reshape(-1) >= 0]
weights = baseline_utils.compute_class_weights(class_labels)
class_weights = torch.tensor(weights, device=device)

loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])

# freeze LLM parameters for warm-up
for name, param in finetuned_model.named_parameters():
     if name.startswith('llm'): 
        param.requires_grad = False

test_losses = []
train_losses = []

for epoch in range(EPOCHS):
    # unfreeze LLM parameters after warm-up
    if epoch > 1:
        for name, param in finetuned_model.named_parameters():
            param.requires_grad = True

    finetuned_model.eval()

    # VALIDATION LOOP
    with torch.no_grad():
        for batch in val_dataloader:

            output = finetuned_model(batch)

            labels = batch['labels'].to(device)
    
            flattened_labels = labels.flatten()

            cbs_logits = output.flatten()[flattened_labels != -100]
            valid_flattened_labels = labels.flatten()[flattened_labels != -100]

            predictions = torch.round(torch.sigmoid(cbs_logits))

            cbs_test_loss =  loss_fn(cbs_logits, valid_flattened_labels)

            test_loss = cbs_test_loss

            test_losses.append(test_loss.cpu().float().detach().numpy())

            # compute metrics on test dataset
            test_acc = baseline_utils.accuracy_fn(y_true=valid_flattened_labels,
                                   y_pred=predictions)
            
            valid_flattened_labels = valid_flattened_labels.cpu().float().numpy()
            predictions = predictions.cpu().float().numpy()
            probabilities = torch.sigmoid(cbs_logits).cpu().float().numpy()

            fpr, tpr, thresholds = metrics.roc_curve(valid_flattened_labels, probabilities)
            roc_auc = metrics.auc(fpr, tpr)

            mcc = metrics.matthews_corrcoef(valid_flattened_labels, predictions)
            f1 = metrics.f1_score(valid_flattened_labels, predictions, average='weighted')

            precision, recall, thresholds = metrics.precision_recall_curve(valid_flattened_labels, probabilities)
            auprc = metrics.auc(recall, precision)

            del labels, cbs_logits, valid_flattened_labels, flattened_labels, probabilities
            gc.collect()
            torch.cuda.empty_cache()
    
    finetuned_model.train()

    batch_losses = []

    # TRAIN

    for batch in train_dataloader:

        output = finetuned_model(batch)

        labels = batch['labels'].to(device)

        flattened_labels = labels.flatten()

        cbs_logits = output.flatten()[flattened_labels != -100]
        valid_flattened_labels = labels.flatten()[flattened_labels != -100]

        loss =  loss_fn(cbs_logits, valid_flattened_labels)
        
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        batch_losses.append(loss.cpu().float().detach().numpy())
        
        del labels, output, cbs_logits, valid_flattened_labels, flattened_labels
        gc.collect()
        torch.cuda.empty_cache()

    train_losses.append(sum(batch_losses) / len(batch_losses))
    print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {test_acc:.2f}% | Test loss: {test_loss:.5f}, AUC: {roc_auc:.4f}, MCC: {mcc:.4f}, F1: {f1:.4f}, AUPRC: {auprc:.4f}, sum: {sum(predictions)}")

finetuned_model.eval()

# VALIDATION LOOP
with torch.no_grad():
    for batch in val_dataloader:

        output = finetuned_model(batch)

        labels = batch['labels'].to(device)

        flattened_labels = labels.flatten()

        cbs_logits = output.flatten()[flattened_labels != -100]
        valid_flattened_labels = labels.flatten()[flattened_labels != -100]

        predictions = torch.round(torch.sigmoid(cbs_logits))
        
        cbs_test_loss =  loss_fn(cbs_logits, valid_flattened_labels)

        test_loss = cbs_test_loss

        test_losses.append(test_loss.cpu().float().detach().numpy())

        # compute metrics on test dataset
        test_acc = baseline_utils.accuracy_fn(y_true=valid_flattened_labels,
                                y_pred=predictions)
        
        valid_flattened_labels = valid_flattened_labels.cpu().float().numpy()
        predictions = predictions.cpu().float().numpy()
        probabilities = torch.sigmoid(cbs_logits).cpu().float().numpy()

        fpr, tpr, thresholds = metrics.roc_curve(valid_flattened_labels, probabilities)
        roc_auc = metrics.auc(fpr, tpr)

        mcc = metrics.matthews_corrcoef(valid_flattened_labels, predictions)

        f1 = metrics.f1_score(valid_flattened_labels, predictions, average='weighted')

        precision, recall, thresholds = metrics.precision_recall_curve(valid_flattened_labels, probabilities)
        auprc = metrics.auc(recall, precision)

        del labels, cbs_logits, valid_flattened_labels, flattened_labels, probabilities
        gc.collect()
        torch.cuda.empty_cache()
        
OUTPUT_PATH = f'{DATA_PATH}/model.pt'
torch.save(finetuned_model, OUTPUT_PATH)

print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {test_acc:.2f}% | Test loss: {test_loss:.5f}, AUC: {roc_auc:.4f}, MCC: {mcc:.4f}, F1: {f1:.4f}, AUPRC: {auprc:.4f}, sum: {sum(predictions)}")
