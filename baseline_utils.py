import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import csv
from torch import nn
from sklearn.utils import class_weight
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

torch.manual_seed(0)

DATASET = 'cryptobench'
DATA_PATH = f'/home/skrhakv/cryptic-nn/data/{DATASET}'
ESM_EMBEDDINGS_PATH = f'{DATA_PATH}/embeddings'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

class SequenceCryptoBenchDataset(Dataset):
    def __init__(self, _Xs, _Ys):
        _Xs_list = []
        _Ys_list = []
        for key, _ in _Xs.items():
            # print(f'Processing {key} ...')
            _Xs_list.append(_Xs[key])
            _Ys_list.append(_Ys[key])

        print('Concatenating ...')
        Xs_list = np.concatenate(_Xs_list, axis=0)
        Ys_list = np.concatenate(_Ys_list, axis=0)

        print('Converting to torch tensor ...')
        self.Xs = torch.tensor(Xs_list, dtype=torch.float32)
        self.Ys = torch.tensor(Ys_list, dtype=torch.int64)

    def __len__(self):
        assert len(self.Xs) == len(self.Ys)
        return len(self.Xs)

    def __getitem__(self, idx):
        x = self.Xs[idx]
        y = self.Ys[idx]
        return x, y

def process_sequence_dataset(annotation_path, embeddings_paths, noncryptic_annotation_path=None, distances_path=None):
    Xs = {}
    Ys = {}

    if noncryptic_annotation_path:
        removed_indices = {}

    if distances_path:
        Ys_distances = {}

    if noncryptic_annotation_path:
        noncryptic_information = {}
        with open(noncryptic_annotation_path) as f:
            reader = csv.reader(f, delimiter=";")
            for row in reader:
                id = row[0].lower() + row[1]
                sequence = row[4]
    
                if row[3] == '':
                    continue
                
                # sanity check
                for (aa, residue_idx) in [(residue[0], int(residue[1:])) for residue in row[3].split(' ')]:
                    assert sequence[residue_idx] == aa
                
                noncryptic_information[id] = (row[3], sequence)

    with open(annotation_path) as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            id = row[0].lower() + row[1]
            sequence = row[4]

            if row[3] == '':
                continue
            
            # load the precomputed embedding
            if id not in Xs:
                for embeddings_path in embeddings_paths:
                    filename = id + '.npy'
                    embedding = np.load(f'{embeddings_path}/{filename}')
                    if id not in Xs:
                        Xs[id] = embedding
                    else:
                        Xs[id] = np.concatenate((Xs[id], embedding), axis = 1)
                    

            # load the annotations denoting whether particular residue is binding or not
            # we use binary annotation: 0=non-binding; 1=binding
            assert id not in Ys
            Ys[id] = np.zeros(embedding.shape[0])
            
            if distances_path:
                distances = np.load(f'{distances_path}/{id}.npy')
                # set the unknown value (-1) to something more reasonable:
                distances[distances == -1] = 0.5
                Ys_distances[id] = distances

                assert len(Ys[id]) == len(Ys_distances[id])

            for (aa, residue_idx) in [(residue[0], int(residue[1:])) for residue in row[3].split(' ')]:
                assert sequence[residue_idx] == aa
                Ys[id][residue_idx] = 1
            
            if noncryptic_annotation_path and id in noncryptic_information:
                # let's remove the non-cryptic residues from the embedding
                noncryptic_residues_indices = []
                for (aa, residue_idx) in [(residue[0], int(residue[1:])) for residue in noncryptic_information[id][0].split(' ')]:

                    assert sequence[residue_idx] == aa
                    assert sequence == noncryptic_information[id][1]

                    # residue is annotated as non-cryptic AND not annotated as cryptic -> residue is non-cryptic
                    if Ys[id][residue_idx] == 0:
                        noncryptic_residues_indices.append(residue_idx)

                # remove the non-cryptic residues from the embedding
                Xs[id] = np.delete(Xs[id], noncryptic_residues_indices, axis=0)
                Ys[id] = np.delete(Ys[id], noncryptic_residues_indices, axis=0)
                removed_indices[id] = noncryptic_residues_indices

    if distances_path:
        return Xs, Ys, Ys_distances
    
    if noncryptic_annotation_path:
        return Xs, Ys, removed_indices

    return Xs, Ys

DECISION_THRESHOLD = 0.95
DROPOUT = 0.3
LAYER_WIDTH = 256
ESM2_DIM  = 2560

class CryptoBenchClassifier(nn.Module):
    def __init__(self, input_dim=ESM2_DIM):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=input_dim, out_features=LAYER_WIDTH)
        self.dropout1 = nn.Dropout(DROPOUT)

        self.layer_2 = nn.Linear(in_features=LAYER_WIDTH, out_features=LAYER_WIDTH)
        self.dropout2 = nn.Dropout(DROPOUT)

        self.layer_3 = nn.Linear(in_features=LAYER_WIDTH, out_features=LAYER_WIDTH)
        self.dropout3 = nn.Dropout(DROPOUT)

        self.layer_4 = nn.Linear(in_features=LAYER_WIDTH, out_features=1)

        self.relu = nn.ReLU()

    def forward(self, x):
      # Intersperse the ReLU activation function between layers
       return self.layer_4(self.dropout3(self.relu(self.layer_3(self.dropout2(self.relu(self.layer_2(self.dropout1(self.relu(self.layer_1(x))))))))))


def compute_class_weights(labels):
    class_weights=class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    class_weights=torch.tensor(class_weights,dtype=torch.float)
    return class_weights

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

torch.manual_seed(42)
        
def train(model, optimizer, epochs, batch_size, train_dataset, test_dataset, decision_threshold=DECISION_THRESHOLD,
          save_aucroc_auprc_to=None):
    model = model.to(device)

    # Create an optimizer
    _, y_train = train_dataset[:]
    X_test, y_test, = test_dataset[:]

    # compute class weights (because the dataset is heavily imbalanced)
    class_weights = compute_class_weights(y_train.numpy()).to(device)

    # BCEWithLogitsLoss - sigmoid is already built-in!
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=8 * class_weights[1])

    X_test, y_test = X_test.to(device), y_test.to(device).float()

    train_losses, test_losses = [], []

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        #
        # TEST
        #
        model.eval()
        with torch.inference_mode():

            test_logits = model(X_test).squeeze()
            # test_pred = torch.round(torch.sigmoid(test_logits))
            test_pred = (torch.sigmoid(test_logits)>decision_threshold).float()

            test_loss = loss_fn(test_logits,
                                y_test)
            test_losses.append(test_loss.cpu().detach().numpy())

            # compute metrics on test dataset
            test_acc = accuracy_fn(y_true=y_test,
                                   y_pred=test_pred)
            
            fpr, tpr, thresholds1 = metrics.roc_curve(y_test.cpu().numpy(), torch.sigmoid(test_logits).cpu().numpy())
            roc_auc = metrics.auc(fpr, tpr)

            mcc = metrics.matthews_corrcoef(y_test.cpu().numpy(), test_pred.cpu().numpy())

            f1 = metrics.f1_score(y_test.cpu().numpy(), test_pred.cpu().numpy(), average='weighted')

            precision, recall, thresholds2 = metrics.precision_recall_curve(y_test.cpu().numpy(), torch.sigmoid(test_logits).cpu().numpy())
            auprc = metrics.auc(recall, precision)


        #
        # TRAIN
        #
        batch_losses = []
        for x_batch, y_batch in train_dataloader:

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).float()

            model.train()

            y_logits = model(x_batch).squeeze()

            loss = loss_fn(y_logits,
                           y_batch)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            batch_losses.append(loss.cpu().detach().numpy())

        train_losses.append(sum(batch_losses) / len(batch_losses))

        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {test_acc:.2f}% | Test loss: {test_loss:.5f}, AUC: {roc_auc:.4f}, MCC: {mcc:.4f}, F1: {f1:.4f}, AUPRC: {auprc:.4f}, sum: {sum(test_pred)}")

    plt.plot(train_losses,label="train loss over epochs")
    plt.plot(test_losses,label="test loss over epochs")
    plt.legend()
    plt.show()

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'r', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    if save_aucroc_auprc_to:
        np.savez(f'/home/skrhakv/cryptic-nn/src/models/auc-auprc/data/{save_aucroc_auprc_to}-rocauc.npz', fpr, tpr, thresholds1)
        np.savez(f'/home/skrhakv/cryptic-nn/src/models/auc-auprc/data/{save_aucroc_auprc_to}-auprc.npz', precision, recall, thresholds2)

