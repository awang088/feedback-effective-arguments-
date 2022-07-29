import numpy as np
import pandas as pd
import os
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torch
import random
from torch.cuda.amp import autocast, GradScaler
import time
from transformers import ElectraForSequenceClassification, get_linear_schedule_with_warmup, AutoTokenizer
from sklearn.metrics import log_loss
import torch.utils.checkpoint
import logging
from contextlib import contextmanager
import sys


# ========================================
# Constants
# ========================================
ex = "002"
TRAIN_PATH = "../../data/train_folds.csv"
if not os.path.exists(f"../output/ex/ex{ex}"):
    os.makedirs(f"../output/ex/ex{ex}")
    os.makedirs(f"../output/ex/ex{ex}/ex{ex}_model")

MODEL_PATH_BASE = f"../output/ex/ex{ex}/ex{ex}_model/ex{ex}"
OOF_SAVE_PATH = f"../output/ex/ex{ex}/ex{ex}_oof.npy"
LOGGER_PATH = f"../output/ex/ex{ex}/ex{ex}.txt"
CONFIG_SAVE_PATH = f"../output/ex/ex{ex}/ex{ex}_model/ex{ex}_config.pth"
TOKENIZER_SAVE_PATH = f"../output/ex/ex{ex}/ex{ex}_tokenizer/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# ===============
# Settings
# ===============
SEED = 42
num_workers = 4
batch_size = 32
num_epoch = 5
max_len = 512
weight_decay = 0.1
lr = 2e-5
warmup_ratio = 0
max_norm = 1.0
folds = 5

model_path = "google/electra-large-discriminator"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.save_pretrained(TOKENIZER_SAVE_PATH)

print_freq = 50

# ===============
# Functions
# ===============
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class FeedbackDataset(Dataset):
    def __init__(self, tokenizer, data_dict, max_len):
        self.tokenizer=tokenizer
        self.max_len=max_len
        self.texts = data_dict['text'].values
        self.labels = data_dict['label'].values
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        SEP = self.tokenizer.sep_token
        text = self.texts[index]
        text = text.replace('[SEP]', SEP)
        tokenized = self.tokenizer(text=self.texts[index],
                                   add_special_tokens=True,
                                   max_length=self.max_len,
                                   padding='max_length',
                                   truncation=True,
                                   return_tensors='pt')
        target = np.zeros(3, dtype=np.float16)
        target[self.labels[index]] = 1
        return tokenized['input_ids'].squeeze(), tokenized['attention_mask'].squeeze(), tokenized['token_type_ids'].squeeze(), target


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9) #
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class FeedbackModel(nn.Module):

    def __init__(self, num_labels=3):
        super(FeedbackModel,self).__init__()
        self.electra = ElectraForSequenceClassification.from_pretrained(
            model_path,
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0,
            summary_last_dropout=0,
            num_labels=num_labels
        )

        self.electra.gradient_checkpointing_enable()
    def forward(self, ids, mask, token_type_ids):
        # pooler
        output = self.electra(ids, attention_mask=mask,
                              token_type_ids=token_type_ids)["logits"]
        return output


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
            self.backup = {}


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def setup_logger(out_file=None, stderr=True, stderr_level=logging.INFO, file_level=logging.DEBUG):
    LOGGER.handlers = []
    LOGGER.setLevel(min(stderr_level, file_level))

    if stderr:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(FORMATTER)
        handler.setLevel(stderr_level)
        LOGGER.addHandler(handler)

    if out_file is not None:
        handler = logging.FileHandler(out_file)
        handler.setFormatter(FORMATTER)
        handler.setLevel(file_level)
        LOGGER.addHandler(handler)

    LOGGER.info("logger set up")
    return LOGGER


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s')


def get_target(x):
    target = np.zeros(3)
    target[x] = 1
    return target

LOGGER = logging.getLogger()
FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
setup_logger(out_file=LOGGER_PATH)


# ================================
# Main
# ================================
data_dict = pd.read_csv(TRAIN_PATH)
y = data_dict['label'].apply(lambda x: get_target(x))
fold_array = data_dict['fold'].values

# ================================
# train
# ================================
with timer('deberta-v3-large'):
    set_seed(SEED)
    oof = np.zeros([len(data_dict), 3])
    for fold in range(folds):
        train_df = data_dict.loc[data_dict['fold'] != fold].reset_index(drop=True)
        valid_df = data_dict.loc[data_dict['fold'] == fold].reset_index(drop=True)

        print(y)
        train_datagen = FeedbackDataset(
            tokenizer, 
            train_df, 
            max_len
        )
        train_generator = DataLoader(
            dataset=train_datagen,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )

        valid_datagen = FeedbackDataset(
            tokenizer, 
            valid_df, 
            max_len
        )
        valid_generator = DataLoader(
            dataset=valid_datagen,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        model = FeedbackModel(num_labels=3)
        # torch.save(model.config, CONFIG_SAVE_PATH)
        model.to(device)

        num_train_steps = int(len(train_df)/batch_size*num_epoch)
        num_warmup_steps = warmup_ratio*num_train_steps

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
            )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=num_warmup_steps, 
            num_training_steps=num_train_steps
        )

        criterion = nn.CrossEntropyLoss()
        best_val = None
        scaler = GradScaler()
        fgm = FGM(model)
        ema = EMA(model, 0.999)
        ema.register()
        for epoch in range(num_epoch):
            with timer(f'model_fold:{epoch}'):
                train_losses = AverageMeter()
                val_losses = AverageMeter()
                val_scores = AverageMeter()
                
                model.train()
                
                for step, (batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_target) in enumerate(train_generator):
                    batch_input_ids = batch_input_ids.to(device)
                    batch_attention_mask = batch_attention_mask.to(device)
                    batch_token_type_ids = batch_token_type_ids.to(device)
                    batch_target = batch_target.to(device)
                    
                    optimizer.zero_grad()

                    with autocast():
                        logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
                        loss = criterion(logits, batch_target) 

                    train_losses.update(loss.item(), logits.size(0))
                    scaler.scale(loss).backward()

                    fgm.attack()
                    logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
                    fgm_loss = criterion(logits, batch_target) 
                    scaler.scale(fgm_loss).backward()
                    fgm.restore()

                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

                    if step % print_freq == 0 or step == (len(train_generator)-1):
                        LOGGER.info(
                            'Epoch: [{0}][{1}/{2}] '
                            'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                            'Grad: {grad_norm:.4f}  '
                            'LR: {lr:.8f}  '
                            .format(
                                epoch+1, 
                                step, 
                                len(train_generator), 
                                loss=train_losses,
                                grad_norm=grad_norm,
                                lr=scheduler.get_last_lr()[0]
                            )
                        )
                
                ema.apply_shadow()
                model.eval()
                preds = np.ndarray([0,3])
                for step, (batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_target) in enumerate(valid_generator):
                    batch_input_ids = batch_input_ids.to(device)
                    batch_attention_mask = batch_attention_mask.to(device)
                    batch_token_type_ids = batch_token_type_ids.to(device)
                    batch_target = torch.from_numpy(np.array(batch_target)).float().to(device)

                    with torch.no_grad():
                        logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
                        loss = criterion(logits, batch_target) 
                    
                    val_losses.update(loss.item(), logits.size(0))
                    logits = softmax(logits.to('cpu').numpy())

                    preds = np.concatenate(
                        [preds, logits], axis=0
                    ) 

                    if step % print_freq == 0 or step == (len(valid_generator)-1):
                        LOGGER.info(
                            'EVAL: [{0}/{1}] '
                            'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                            .format(
                                step, 
                                len(valid_generator),
                                loss=val_losses,
                            )
                        )
                
                ema.restore()
                # ===================
                # early stop
                # ===================
                if not best_val:
                    best_val = val_losses.avg
                    oof[fold_array == fold] = preds
                    torch.save(model.state_dict(), MODEL_PATH_BASE +
                               f"_{fold}.pth")  # Saving the model
                    continue

                if val_losses.avg <= best_val:
                    best_val = val_losses.avg
                    oof[fold_array == fold] = preds
                    torch.save(model.state_dict(), MODEL_PATH_BASE +
                               f"_{fold}.pth")  # Saving current best model

y = data_dict['label'].values
val_score = log_loss(oof, y)
LOGGER.info(f'oof_score:{val_score}')
np.save(OOF_SAVE_PATH, oof)
