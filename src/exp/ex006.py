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
from transformers import get_linear_schedule_with_warmup, RobertaTokenizer, RobertaModel
from sklearn.metrics import log_loss
import torch.utils.checkpoint
import logging
from contextlib import contextmanager
import sys


# ========================================
# Constants
# ========================================
ex = "006"
TRAIN_PATH = "../../data/train_folds.csv"
if not os.path.exists(f"../output/ex/ex{ex}"):
    os.makedirs(f"../output/ex/ex{ex}")
    os.makedirs(f"../output/ex/ex{ex}/ex{ex}_model")

MODEL_PATH_BASE = f"../output/ex/ex{ex}/ex{ex}_model/ex{ex}"
OOF_SAVE_PATH = f"../output/ex/ex{ex}/ex{ex}_oof.npy"
LOGGER_PATH = f"../output/ex/ex{ex}/ex{ex}.txt"
TOKENIZER_SAVE_PATH = f"../output/ex/ex{ex}/ex{ex}_tokenizer/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# ===============
# Settings
# ===============
SEED = 42
num_workers = 4
batch_size = 32
num_epoch = 4
max_len = 512
weight_decay = 0.1
lr = 2e-5
warmup_ratio = 0
max_norm = 1.0
folds = 5

MODEL_PATH = "roberta-large"
tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
tokenizer.save_pretrained(TOKENIZER_SAVE_PATH)

print_freq = 50
use_awp = False

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
        tokenized = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )
        target = np.zeros(3, dtype=np.float16)
        target[self.labels[index]] = 1
        return tokenized['input_ids'].squeeze(), tokenized['attention_mask'].squeeze(), tokenized['token_type_ids'].squeeze(), target


class FeedbackModel(nn.Module):

    def __init__(self):
        super(FeedbackModel,self).__init__()
        self.roberta = RobertaModel.from_pretrained(
            MODEL_PATH,
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0
        )

        # self.dropout = nn.Dropout(p=0.2)
        self.ln = nn.LayerNorm(1024)
        self.out = nn.Linear(1024, 3)
        self.roberta.gradient_checkpointing_enable()

    def forward(self, ids, mask, token_type_ids):
        # pooler
        emb = self.roberta(ids, attention_mask=mask, token_type_ids=token_type_ids)[
            "last_hidden_state"]
        emb = torch.mean(emb, axis=1)
        output = self.ln(emb)
        # output = self.dropout(output)
        output = self.out(output)
        return output


class AWP:
    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=1,
        adv_eps=0.2,
        start_epoch=0,
        adv_step=1,
        scaler=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.adv_step = adv_step
        self.backup = {}
        self.backup_eps = {}
        self.scaler = scaler

    def attack_backward(self, x, y, attention_mask, token_type_ids, epoch):
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return None

        self._save() 
        for i in range(self.adv_step):
            self._attack_step() 
            with torch.cuda.amp.autocast():
                tr_logits = self.model(x, attention_mask, token_type_ids)
                adv_loss = nn.CrossEntropyLoss()(tr_logits, y)
                adv_loss = adv_loss.mean()
            self.optimizer.zero_grad()
            self.scaler.scale(adv_loss).backward()
            
        self._restore()

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )
                # param.data.clamp_(*self.backup_eps[name])

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


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
    torch.backends.cudnn.benchmark = False


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
with timer('roberta_large'):
    set_seed(SEED)
    oof = np.zeros([len(data_dict), 3])
    for fold in range(folds):
        train_df = data_dict.loc[data_dict['fold'] != fold].reset_index(drop=True)
        valid_df = data_dict.loc[data_dict['fold'] == fold].reset_index(drop=True)

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
        
        model = FeedbackModel()
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
            betas=(0.9, 0.98),
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
        awp = AWP(
            model,
            optimizer,
            adv_lr=0.0005,
            adv_eps=0.001,
            start_epoch=1,
            scaler=scaler
        )
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
                    if use_awp and train_losses.avg < 0.7:
                        awp.attack_backward(batch_input_ids, batch_target, batch_attention_mask, batch_token_type_ids, epoch)
                    
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
                
                model.eval()
                preds = np.ndarray([0,3])
                for step, (batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_target) in enumerate(valid_generator):
                    batch_input_ids = batch_input_ids.to(device)
                    batch_attention_mask = batch_attention_mask.to(device)
                    batch_token_type_ids = batch_token_type_ids.to(device)
                    batch_target = batch_target.to(device)

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
