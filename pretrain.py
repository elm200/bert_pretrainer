from typing import Any
from pathlib import Path
import json
import random
import logging
from datetime import datetime

import torch
from torch import nn
from transformers import (
    BertConfig,
    BertJapaneseTokenizer,
    AdamW,
    BertForPreTraining,
)
from modeling_bert import BertForPreTraining as BertForPreTrainingWithoutNSP
import pandas as pd
import numpy as np


torch.backends.cudnn.enabled = False

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s:%(levelname)s %(message)s", level=logging.INFO)

PRJ_ROOT = Path(__file__).parent
TRAIN_PATH = PRJ_ROOT / "data" / "wikipedia.csv"
MAX_SEQ_LEN = 64
MASK_ID = 4
IGNORE_ID = -100
BATCH_SIZE = 16
MAX_STEPS = 150000

tokenizer = BertJapaneseTokenizer.from_pretrained(
    "cl-tohoku/bert-base-japanese-whole-word-masking"
)


def create_sent_pairs(
    sents_list: list[str], batch_size: int, seq_pair_ratio: float = 0.5
) -> list[tuple[str, str, int]]:
    n_seq_pair = int(batch_size * seq_pair_ratio)
    n_random_pair = batch_size - n_seq_pair
    pairs = []

    for i in range(n_seq_pair):
        while True:
            sents = json.loads(random.choice(sents_list))
            if len(sents) >= 2:
                break
        st = random.randint(0, len(sents) - 2) if len(sents) > 2 else 0
        pair = tuple(sents[st : st + 2]) + (0,)
        pairs.append(pair)

    for i in range(n_random_pair):
        ss1, ss2 = random.sample(sents_list, 2)
        sents1 = json.loads(ss1)
        sents2 = json.loads(ss2)
        pair = (random.choice(sents1), random.choice(sents2), 1)
        pairs.append(pair)

    pairs = random.sample(pairs, len(pairs))
    return pairs


def mask_token_ids_and_labels(
    token_ids: list[int], mask_rate: float = 0.15
) -> tuple[list[int], list[int]]:
    indexes = []
    for i in range(len(token_ids)):
        r = random.random()
        if r < mask_rate:
            indexes.append(i)
    labels = [IGNORE_ID] * len(token_ids)
    if len(indexes) == 0:
        return token_ids, labels
    labels = np.array(labels)
    indexes = np.array(indexes)
    token_ids = np.array(token_ids)
    masked_token_ids = np.array(token_ids)
    labels[indexes] = token_ids[indexes]
    masked_token_ids[indexes] = MASK_ID
    for i in indexes:
        r = random.random()
        if r < 0.1:
            masked_token_ids[i] = token_ids[i]
        elif r < 0.2:
            masked_token_ids[i] = random.choice(token_ids)
    return masked_token_ids.tolist(), labels.tolist()


def encode_sent_pairs(sent_pairs: list[tuple[str, str, int]]) -> dict[str, Any]:
    next_sentence_labels = []
    tokens_pairs = []
    labels_pairs = []
    for sent1, sent2, next_sentence_label in sent_pairs:
        next_sentence_labels.append(next_sentence_label)

        tokens1 = tokenizer.tokenize(sent1)
        token_ids1 = tokenizer.convert_tokens_to_ids(tokens1)
        token_ids1, labels1 = mask_token_ids_and_labels(token_ids1)

        tokens2 = tokenizer.tokenize(sent2)
        token_ids2 = tokenizer.convert_tokens_to_ids(tokens2)
        token_ids2, labels2 = mask_token_ids_and_labels(token_ids2)

        tokens_pairs.append((token_ids1, token_ids2))
        labels_pairs.append((labels1, labels2))

    encoded = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=tokens_pairs,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
        truncation=True,
    )

    encoded_label = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=labels_pairs,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
        truncation=True,
    )

    encoded = dict(encoded)
    encoded["labels"] = encoded_label["input_ids"]
    encoded["next_sentence_label"] = torch.tensor(next_sentence_labels)

    return encoded


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    output_base_dir = PRJ_ROOT / "output" / datetime.now().strftime("train%Y%m%d%H%M%S")
    output_base_dir.mkdir(exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("device: {}".format(device))

    df_train = pd.read_csv(TRAIN_PATH)
    df_train.columns = ["doc_id", "sents"]
    sents_list = df_train["sents"].values.tolist()
    logger.info("len(sents_list): {}".format(len(sents_list)))

    config = BertConfig()
    config.num_hidden_layers = 3
    config.num_attention_heads = 12
    config.hidden_size = 768
    config.intermediate_size = 3072
    config.max_position_embeddings = 512
    config.vocab_size = 32000
    model = BertForPreTrainingWithoutNSP(config)
    model.to(device)

    logger.info(config)
    logger.info(model)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    model.train()
    n_batch = 0
    train_losses = []
    for i in range(1, MAX_STEPS + 1):
        optimizer.zero_grad()
        sent_pairs = create_sent_pairs(sents_list, batch_size=BATCH_SIZE)
        encoded = encode_sent_pairs(sent_pairs)
        res = model(
            encoded["input_ids"].to(device),
            token_type_ids=None,
            attention_mask=encoded["attention_mask"].to(device),
            labels=encoded["labels"].to(device),
            next_sentence_label=encoded["next_sentence_label"].to(device),
        )
        loss = res.loss
        if i % 100 == 0:
            logger.info("training step {}, loss {}".format(i, loss))
            train_losses.append((i, loss.item()))
            df_train_loss = pd.DataFrame(train_losses, columns=["step", "train_loss"])
            df_train_loss.to_csv(output_base_dir / "train_loss.csv", index=False)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if i == 1 or i % 50000 == 0:
            save_dir = output_base_dir / "step{}".format(i)
            model.save_pretrained(save_directory=save_dir)
