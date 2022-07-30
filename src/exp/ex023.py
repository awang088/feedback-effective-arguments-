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
from transformers import get_linear_schedule_with_warmup, DebertaV2TokenizerFast, DebertaV2Model, DebertaV2PreTrainedModel, DebertaV2Config
from transformers.models.deberta.modeling_deberta import ContextPooler, StableDropout
from sklearn.metrics import log_loss
import torch.utils.checkpoint
import logging
from contextlib import contextmanager
import sys


# ========================================
# Constants
# ========================================
ex = "023"
TRAIN_PATH = "../../data/train_folds.csv"
if not os.path.exists(f"../output/ex/ex{ex}"):
    os.makedirs(f"../output/ex/ex{ex}")
    os.makedirs(f"../output/ex/ex{ex}/ex{ex}_model")

MODEL_PATH_BASE = f"../output/ex/ex{ex}/ex{ex}_model/ex{ex}"
OOF_SAVE_PATH = f"../output/ex/ex{ex}/ex{ex}_oof.npy"
LOGGER_PATH = f"../output/ex/ex{ex}/ex{ex}.txt"
CONFIG_SAVE_PATH = f"../output/ex/ex{ex}/ex{ex}_model/ex{ex}_config.pth"
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
weight_decay = 0.01
encoder_lr = 2e-5
decoder_lr = 1e-3
warmup_ratio = 0
folds = 5
max_norm = 1.0

MODEL_PATH = "microsoft/deberta-v3-large"
tokenizer = DebertaV2TokenizerFast.from_pretrained(MODEL_PATH)
tokenizer.save_pretrained(TOKENIZER_SAVE_PATH)

print_freq = 50
use_awp = True

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
        target = self.labels[index]
        return tokenized['input_ids'].squeeze(), tokenized['attention_mask'].squeeze(), tokenized['token_type_ids'].squeeze(), target


class FeedbackModel(DebertaV2PreTrainedModel):

    def __init__(self, config):
        super(FeedbackModel,self).__init__(config)
        self.deberta = DebertaV2Model(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = nn.Linear(output_dim, 3)
        self.dropout1 = StableDropout(0.1)
        self.dropout2 = StableDropout(0.2)
        self.dropout3 = StableDropout(0.3)
        self.dropout4 = StableDropout(0.4)
        self.dropout5 = StableDropout(0.5)

        self.bilstm = nn.LSTM(
            config.hidden_size, 
            (config.hidden_size) // 2, 
            num_layers=2, 
            dropout=0.1, 
            batch_first=True,
            bidirectional=True
        )

        self.deberta.gradient_checkpointing_enable()
        self.init_weights()
    def forward(self, ids, mask, token_type_ids=None):
        # pooler
        outputs = self.deberta(ids, attention_mask=mask, token_type_ids=token_type_ids)
        encoder_layer = outputs[0]
        encoder_layer, _ = self.bilstm(encoder_layer)

        pooled_output = self.pooler(encoder_layer)

        preds1 = self.classifier(self.dropout1(pooled_output))
        preds2 = self.classifier(self.dropout2(pooled_output))
        preds3 = self.classifier(self.dropout3(pooled_output))
        preds4 = self.classifier(self.dropout4(pooled_output))
        preds5 = self.classifier(self.dropout5(pooled_output))
        logits = (preds1 + preds2 + preds3 + preds4 + preds5) / 5

        return logits



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


LOGGER = logging.getLogger()
FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
setup_logger(out_file=LOGGER_PATH)


# ================================
# Main
# ================================
data_dict = pd.read_csv(TRAIN_PATH)
y = data_dict['label'].values
fold_array = data_dict['fold'].values

# ================================
# train
# ================================
with timer('deberta_v3_large'):
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
            batch_size=batch_size*2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        config = DebertaV2Config.from_pretrained(MODEL_PATH)
        torch.save(config, CONFIG_SAVE_PATH)
        model = FeedbackModel.from_pretrained(MODEL_PATH, config=config)
        model.to(device)

        num_train_steps = int(len(train_df)/batch_size*num_epoch)
        num_warmup_steps = warmup_ratio*num_train_steps

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.deberta.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.deberta.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "deberta" not in n],
             'lr': decoder_lr, 'weight_decay': 0.0}
        ]

        optimizer = optim.AdamW(
            optimizer_parameters,
            lr=encoder_lr,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
            eps=1e-6
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
            adv_lr=1.0,
            adv_eps=0.01,
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

                    with autocast():
                        logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
                        loss = criterion(logits, batch_target) 

                    train_losses.update(loss.item(), logits.size(0))

                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    if use_awp and train_losses.avg < 0.65:
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
                        print(logits[0], batch_target[0])
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

val_score = log_loss(oof, y)
LOGGER.info(f'oof_score:{val_score}')
np.save(OOF_SAVE_PATH, oof)
