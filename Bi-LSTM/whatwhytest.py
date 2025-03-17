import torch
from datasets import load_dataset  # hugging-face dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import torch.optim as optim
from torch.nn.functional import one_hot
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torchmetrics.functional import accuracy, recall, precision, f1_score
import os
from pathlib import Path

torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.set_float32_matmul_precision('high')

CHECKPOINT_PATH = Path('./checkpoints/bi-lstm.ckpt')

batch_size = 64
epochs = 30
dropout = 0.4
rnn_hidden = 768
rnn_layer = 1
class_num = 4
lr = 0.001

token = BertTokenizer.from_pretrained('bert-base-uncased')

# todo: Customized datasets
import json

class MydataSet(Dataset):
    def __init__(self, path, split):
        self.dataset = load_dataset('csv', data_files=path, split=split)
    def __getitem__(self, item):
        text = self.dataset[item]['new_message1']
        label = self.dataset[item]['label']
        return text, label
    def __len__(self):
        return len(self.dataset)


# todo: Defining batch functions
def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]
    # Combine words and encode
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,  # Individual sentences are involved in encoding
        truncation=True,  # Truncate when sentence length is greater than max_length
        padding='max_length',  # Always pad to max_length
        max_length=200,
        return_tensors='pt',  # Return in the form of pytorch, can take the value of tf, pt, np, the default is to return the list
        return_length=True,
    )

    # input_ids: number after encoding
    # attention_mask: The position that is zero complementary is 0, and the other positions are 1
    input_ids = data['input_ids']  # input_ids are the encoded words
    attention_mask = data['attention_mask']  # The pad position is 0, the other positions are 1
    token_type_ids = data['token_type_ids']  # (In the case of a pair of sentences) the position of the first sentence and the special symbol is 0, the position of the second sentence is 1
    labels = torch.LongTensor(labels)  # Labels from this batch

    # print(data['length'], data['length'].max())
    return input_ids, attention_mask, token_type_ids, labels


# todo: Define the model, use bert pre-training for upstream, choose a bi-directional LSTM model for downstream tasks, and finally add a fully connected layer
class BiLSTMClassifier(nn.Module):
    def __init__(self, drop, hidden_dim, output_dim):
        super(BiLSTMClassifier, self).__init__()
        self.drop = drop
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Load bert Chinese model and generate embedding layer
        self.embedding = BertModel.from_pretrained('bert-base-uncased')
        # Remove and move to gpu
        # Freeze upstream model parameters (no pre-training model parameter learning)
        for param in self.embedding.parameters():
            param.requires_grad_(False)
        # Generate downstream RNN layers as well as fully connected layers
        self.lstm = nn.LSTM(input_size=768, hidden_size=self.hidden_dim, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=self.drop)
        self.fc = nn.Linear(self.hidden_dim * 2, self.output_dim)
        # No activation is required when using CrossEntropyLoss as a loss function. Because actually CrossEntropyLoss implements softmax-log-NLLLoss together

    def forward(self, input_ids, attention_mask, token_type_ids):
        embedded = self.embedding(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embedded = embedded.last_hidden_state  # The 0th dimension is what we need for embedding, embedding.last_hidden_state = embedding[0]
        out, (h_n, c_n) = self.lstm(embedded)
        output = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        output = self.fc(output)
        return output


# todo: Define pytorch lightning
class BiLSTMLighting(pl.LightningModule):
    def __init__(self, drop, hidden_dim, output_dim):
        super(BiLSTMLighting, self).__init__()
        self.model = BiLSTMClassifier(drop, hidden_dim, output_dim)  # Setting up the model
        self.criterion = nn.CrossEntropyLoss()  # Setting the loss function
        self.train_dataset = MydataSet('./data/archive/train_clean.csv', 'train')
        self.val_dataset = MydataSet('./data/archive/val_clean.csv', 'train')
        self.test_dataset = MydataSet('./data/archive/test_clean.csv', 'train')
        self.test_step_outputs = []  # Initialize list to store test outputs

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=lr)
        return optimizer

    def forward(self, input_ids, attention_mask, token_type_ids):  # forward(self,x)
        return self.model(input_ids, attention_mask, token_type_ids)

    def train_dataloader(self):
        train_loader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, collate_fn=collate_fn,
                                  shuffle=True)
        return train_loader

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch  # x, y = batch
        y = one_hot(labels, num_classes=4)
        # Convert one_hot_labels type to float
        y = y.to(dtype=torch.float)
        # forward pass
        y_hat = self.model(input_ids, attention_mask, token_type_ids)
        y_hat = y_hat.squeeze()  # Squeeze [128, 1, 3] into [128,3]
        loss = self.criterion(y_hat, y)  # criterion(input, target)
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)  # Output the loss on the console
        return loss  # The log must be returned to be useful

    def val_dataloader(self):
        val_loader = DataLoader(dataset=self.val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
        return val_loader

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        y = one_hot(labels, num_classes=4)
        y = y.to(dtype=torch.float)
        # forward pass
        y_hat = self.model(input_ids, attention_mask, token_type_ids)
        y_hat = y_hat.squeeze()
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return loss

    def test_dataloader(self):
        test_loader = DataLoader(dataset=self.test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
        return test_loader

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        # Forward propagation
        y_hat = self.model(input_ids, attention_mask, token_type_ids)
        # Prediction
        pred = torch.argmax(y_hat, dim=1)
        
        output = {"pred": pred, "label": labels}
        self.test_step_outputs.append(output)  # Store output
        return output

    def on_test_epoch_end(self):
        # Access the collected outputs via self.test_step_outputs
        all_predictions = []
        all_labels = []
        
        for output in self.test_step_outputs:
            preds = output["pred"].cpu()
            labels = output["label"].cpu()
            
            all_predictions.append(preds)
            all_labels.append(labels)
        
        # Concatenate all batch predictions and labels
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        
        # Calculate metrics
        accuracy_score = (all_predictions == all_labels).float().mean()
        precision_score = precision(all_predictions, all_labels, task="multiclass", num_classes=class_num, average='macro')
        recall_score = recall(all_predictions, all_labels, task="multiclass", num_classes=class_num, average='macro')
        f1 = f1_score(all_predictions, all_labels, task="multiclass", num_classes=class_num, average='macro')
        
        # Log metrics
        self.log("test_accuracy", accuracy_score)
        self.log("test_precision", precision_score)
        self.log("test_recall", recall_score)
        self.log("test_f1", f1)
        
        # Print metrics
        print(f"Test Results:")
        print(f"Accuracy: {accuracy_score:.4f}")
        print(f"Precision: {precision_score:.4f}")
        print(f"Recall: {recall_score:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Per-class metrics
        class_precision = precision(all_predictions, all_labels, task="multiclass", num_classes=class_num, average=None)
        class_recall = recall(all_predictions, all_labels, task="multiclass", num_classes=class_num, average=None)
        class_f1 = f1_score(all_predictions, all_labels, task="multiclass", num_classes=class_num, average=None)
        
        print("\nPer-class metrics:")
        for i in range(class_num):
            print(f"Class {i}: Precision={class_precision[i]:.4f}, Recall={class_recall[i]:.4f}, F1={class_f1[i]:.4f}")
        
        # Save predictions to file
        with open("predictions.txt", "w") as f:
            f.write(f"Overall Metrics: Accuracy={accuracy_score:.4f}, Precision={precision_score:.4f}, Recall={recall_score:.4f}, F1={f1:.4f}\n\n")
            f.write("Predictions:\n")
            for pred, label in zip(all_predictions.tolist(), all_labels.tolist()):
                f.write(f"Predicted: {pred}, Actual: {label}\n")
        
        print(f"Test completed. Predictions and metrics saved to predictions.txt")
        
        # Clear the outputs to free memory
        self.test_step_outputs.clear()


def test():
    # Load the parameters of the previously trained optimal model
    model = BiLSTMLighting.load_from_checkpoint(checkpoint_path=CHECKPOINT_PATH,
                                               drop=dropout, hidden_dim=rnn_hidden, output_dim=class_num)
    trainer = Trainer(fast_dev_run=False)
    result = trainer.test(model, verbose=True)
    print("Test completed with result:", result)

if __name__ == '__main__':
    test()
