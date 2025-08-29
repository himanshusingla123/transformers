## 🔹 1. Random Masking – kya hota hai?

* Masked Language Modeling (MLM) me model ko randomly kuch tokens (words/subwords) **hide** karke train karte hain.
* Example:
  `"The cat sat on the mat"` → `"The [MASK] sat on the mat"`.
* Model ka task: predict kare ki `[MASK] = "cat"`.

👉 **Random masking effective kyu hai?**

* Model har training step pe alag-alag jagah mask dekhta hai → better generalization.
* Agar hamesha same jagah mask karte, toh model overfit ho jata.
* Yeh BERT jaise models ke pretraining ka base idea hai.

---

## 🔹 2. Tumhare code me kya ho raha hai?

```python
samples = [lm_datasets["train"][i] for i in range(2)]
```

👉 Yeh dataset se 2 training samples nikal raha hai.
(`lm_datasets["train"]` ek tokenized dataset hai.)

---

```python
for sample in samples:
    _ = sample.pop("word_ids")
```

👉 Har sample ke andar ek extra key `"word_ids"` hoti hai (jo batati hai ki kaunsa token kis original word se belong karta hai).
Isko hata diya gaya hai kyunki MLM ke liye zaroori nahi hai.

---

```python
for chunk in data_collator(samples)["input_ids"]:
    print(f"\n'>>> {tokenizer.decode(chunk)}'")
```

👉 Ye **data\_collator** ko call karta hai.

* `data_collator` = `DataCollatorForLanguageModeling` (Hugging Face ka).
* Iska kaam hai:

  1. Randomly kuch tokens ko `[MASK]` me replace karna (MLM ke liye).
  2. `input_ids`, `labels` etc. ka batch banana.

👉 Phir `["input_ids"]` liya ja raha hai → jo mask lagne ke baad ke token IDs hain.
👉 `tokenizer.decode(chunk)` → IDs ko wapas readable text me convert karta hai.

---

## 🔹 3. Example (Illustration)

Suppose original sample tha:

```text
"I love playing football"
```

Random masking ke baad collator output kar sakta hai:

```text
"I [MASK] playing football"
```

Ya kabhi:

```text
"I love [MASK] football"
```

Har run me alag hoga (isi liye “random masking”).

---

## 🔹 4. Is code ka matlab

Tumhara code basically **2 samples uthata hai → word\_ids hataata hai → random masking apply karta hai → masked input ko decode karke print karta hai**.

Output kuch aisa milega:

```
>>> I [MASK] playing football
>>> The cat sat [MASK] the mat
```

---

✅ **Conclusion**:
Ye code dikhata hai ki **data\_collator randomly tokens ko mask karke input\_ids bana raha hai** jo model ko MLM training ke liye feed kiya jaayega.

---

Bahut accha question pucha aapne 👌. Yeh poora code ek **Masked Language Model (MLM)** ko fine-tune karne ka workflow dikhata hai 🤗 Transformers, 🤗 Accelerate aur Hugging Face Hub ke sath. Main har ek line ka matlab Romanized Hindi me breakdown karta hoon, aur jaha better approach ho sakti hai, uska bhi point karunga:

---

## 🔹 Part 1: Random Masking Fix for Evaluation

```python
def insert_random_mask(batch):
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = data_collator(features)
    # Create a new "masked" column for each column in the dataset
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}
```

👉 **Matlab**:

* Har batch ke upar **random masking** lagaya ja raha hai. Normally `DataCollatorForLanguageModeling` har baar random masking karta hai → isse evaluation ke time **perplexity fluctuate** hoti hai.
* Is function me batch ke tokens lekar, unhe collator se mask kiya aur naye columns banaye jaise `"masked_input_ids"`, `"masked_labels"`.

📌 **Better kya karna chahiye?**
Aap evaluation ke liye **ek fixed mask** bana lo taaki har run me results stable ho. Yeh hi yaha implement ho raha hai.

---

```python
downsampled_dataset = downsampled_dataset.remove_columns(["word_ids"])
```

👉 "word\_ids" column hata diya gaya hai (jo token-to-word mapping rakhta tha). Masking ke baad wo kaam ka nahi hai.

---

```python
eval_dataset = downsampled_dataset["test"].map(
    insert_random_mask,
    batched=True,
    remove_columns=downsampled_dataset["test"].column_names,
)
```

👉 Test set par random masking apply karke **new masked dataset** banaya. Purane columns hata diye aur unki jagah masked wale aaye.

---

```python
eval_dataset = eval_dataset.rename_columns(
    {
        "masked_input_ids": "input_ids",
        "masked_attention_mask": "attention_mask",
        "masked_labels": "labels",
    }
)
```

👉 Column names ko **standard names** me badal diya jisse model samajh sake (`input_ids`, `attention_mask`, `labels`).

---

## 🔹 Part 2: Dataloaders

```python
from torch.utils.data import DataLoader
from transformers import default_data_collator

batch_size = 64
train_dataloader = DataLoader(
    downsampled_dataset["train"],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator,
)
```

👉 Training data loader bana, shuffle kiya aur collator diya jo **random masking har batch par apply karega**.

```python
eval_dataloader = DataLoader(
    eval_dataset, batch_size=batch_size, collate_fn=default_data_collator
)
```

👉 Evaluation ke liye **default collator** use kiya (kyunki masking pehle se ho chuki hai).

---

## 🔹 Part 3: Model & Optimizer

```python
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
```

👉 Pretrained MLM model load kiya (jaise `bert-base-uncased`, `distilbert-base-uncased`, etc).

```python
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)
```

👉 Optimizer AdamW lagaya (standard NLP fine-tuning optimizer). Learning rate `5e-5` default safe value hai.

📌 **Better option:** Agar dataset bada hai to `lr=3e-5` ya scheduler ke sath warmup steps lena zyada stable hota hai.

---

## 🔹 Part 4: Accelerator (Multi-GPU/TPU Friendly Training)

```python
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
```

👉 🤗 Accelerate ka `Accelerator` class use kiya gaya hai jisse training CPU/GPU/TPU me seamless chale.
`accelerator.prepare` sabko wrap karta hai → correct device aur distributed training handle karta hai.

📌 Yeh best practice hai, kyunki manually `.cuda()` ya DDP setup karne ki zarurat nahi hoti.

---

## 🔹 Part 5: Scheduler (Learning Rate Control)

```python
from transformers import get_scheduler

num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
```

👉 Learning rate scheduler banaya gaya hai jo training ke dauraan learning rate ko **linearly decrease** karega.
Warmup steps = `0` hai, lekin better hota agar thoda warmup dete (`num_warmup_steps = 500` jaise).

📌 **Tip**: Warmup steps → model pehle thoda dhire se seekhe, phir stable training kare. Isse training zyada smooth hoti hai.

---

## 🔹 Part 6: Hugging Face Hub Push

```python
from huggingface_hub import get_full_repo_name

model_name = "distilbert-base-uncased-finetuned-imdb-accelerate"
repo_name = get_full_repo_name(model_name)
repo_name
```

👉 Hugging Face Hub par ek repo ka naam generate kiya ja raha hai (auto format: `username/model_name`).

```python
from huggingface_hub import Repository

output_dir = model_name
repo = Repository(output_dir, clone_from=repo_name)
```

👉 Local folder me HF repo clone karke model outputs waha save honge. Phir push karna easy ho jata hai.

---

✅ **Summary (Best Practices for You):**

1. Random masking fix karna evaluation ke liye achha hai (stable results ke liye).
2. Warmup steps > 0 rakhna better hoga.
3. Agar dataset bada hai → learning rate thoda kam (2e-5 ya 3e-5) rakhna.
4. Accelerator use karna best hai multi-GPU/TPU training ke liye.
5. HF Hub par model push karne se share/restore easy ho jata hai.

---
Awesome — here’s a clean, **end-to-end, ready-to-run** script that fine-tunes a **Masked Language Model (MLM)** with:

* 🤗 Datasets (`wikitext` tiny split for speed)
* 🤗 Transformers (`BERT` MLM head)
* A **stable evaluation** (masking is done once for the eval set)
* 🤗 Accelerate (GPU / mixed-precision / multi-GPU ready)
* Linear LR scheduler **with warmup**
* Optional push to the 🤗 Hub

Just paste this into a Python file (e.g., `train_mlm_accelerate.py`) and run.

```python
# ============================================================
# End-to-end MLM fine-tuning with stable evaluation masking
# ============================================================
# pip install -U datasets transformers accelerate huggingface_hub
# Optional (tiny dataset): pip install -U evaluate

import math
import os
import random
import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    default_data_collator,
    get_scheduler,
)
from torch.utils.data import DataLoader
from accelerate import Accelerator
from huggingface_hub import get_full_repo_name, Repository

# --------------------------
# Config
# --------------------------
MODEL_CHECKPOINT = "bert-base-uncased"
BLOCK_SIZE = 128                 # sequence length after chunking
MLM_PROB = 0.15                  # masking probability
BATCH_SIZE = 32                  # per device
NUM_EPOCHS = 3
LEARNING_RATE = 3e-5
WARMUP_RATIO = 0.06              # ~6% warmup is a good default
SEED = 42
PUSH_TO_HUB = False              # set True if you want to push
MODEL_NAME = "bert-mlm-wikitext-accel-stable-eval"

# --------------------------
# Reproducibility
# --------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --------------------------
# 1) Load a small text dataset
#    (wikitext-2-raw-v1 is tiny; perfect for a demo)
# --------------------------
raw = load_dataset("wikitext", "wikitext-2-raw-v1")
# We'll use train for training; validation for evaluation
# (You can also concatenate splits or switch to a larger corpus.)

# --------------------------
# 2) Tokenizer and tokenization
#    We'll do "line-by-line": keep short lines; then concatenate & chunk.
# --------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=True)

def tokenize_fn(batch):
    return tokenizer(batch["text"], return_special_tokens_mask=True, truncation=False)

tokenized = raw.map(tokenize_fn, batched=True, remove_columns=["text"])

# Concatenate texts and split into fixed-length blocks
def group_texts(examples):
    # Concatenate each field
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_len = len(concatenated["input_ids"])
    # Drop the small remainder to keep shapes aligned
    total_len = (total_len // BLOCK_SIZE) * BLOCK_SIZE
    result = {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_len, BLOCK_SIZE)]
        for k, t in concatenated.items()
    }
    return result

lm_datasets = tokenized.map(group_texts, batched=True)

# --------------------------
# 3) Data collators
#    - TRAIN: standard MLM collator (random masking each batch)
#    - EVAL:  we'll pre-mask ONCE and then use default collator
# --------------------------
train_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm_probability=MLM_PROB
)

# Pre-mask once for stable evaluation
def insert_random_mask(batch):
    """
    Convert a batch dict of lists (columnar) into a list of feature dicts,
    let DataCollatorForLanguageModeling apply masking, then return
    masked_* numpy arrays as new columns.
    """
    # Convert columnar -> row-wise features for the collator
    features = [dict(zip(batch, values)) for values in zip(*batch.values())]
    masked = train_collator(features)  # applies masking once
    return {"masked_" + k: v.numpy() for k, v in masked.items()}

eval_ds = lm_datasets["validation"].map(
    insert_random_mask,
    batched=True,
    remove_columns=lm_datasets["validation"].column_names,
)

# Rename masked_* to standard names expected by the model
eval_ds = eval_ds.rename_columns(
    {
        "masked_input_ids": "input_ids",
        "masked_attention_mask": "attention_mask",
        "masked_labels": "labels",
    }
)

# --------------------------
# 4) Dataloaders
# --------------------------
train_loader = DataLoader(
    lm_datasets["train"],
    shuffle=True,
    batch_size=BATCH_SIZE,
    collate_fn=train_collator,     # random masking on-the-fly
)
eval_loader = DataLoader(
    eval_ds,
    shuffle=False,
    batch_size=BATCH_SIZE,
    collate_fn=default_data_collator,  # already masked
)

# --------------------------
# 5) Model, optimizer, scheduler, accelerator
# --------------------------
model = AutoModelForMaskedLM.from_pretrained(MODEL_CHECKPOINT)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

accelerator = Accelerator()  # mixed-precision/distributed if available
model, optimizer, train_loader, eval_loader = accelerator.prepare(
    model, optimizer, train_loader, eval_loader
)

num_update_steps_per_epoch = len(train_loader)
num_training_steps = NUM_EPOCHS * num_update_steps_per_epoch
num_warmup_steps = int(WARMUP_RATIO * num_training_steps)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)

# --------------------------
# 6) (Optional) Prepare a Hub repo to save checkpoints
# --------------------------
if PUSH_TO_HUB:
    repo_name = get_full_repo_name(MODEL_NAME)
    output_dir = MODEL_NAME
    repo = Repository(local_dir=output_dir, clone_from=repo_name, token=True)
else:
    output_dir = MODEL_NAME
os.makedirs(output_dir, exist_ok=True)

# --------------------------
# 7) Training & evaluation loops
# --------------------------
best_eval_loss = float("inf")

for epoch in range(NUM_EPOCHS):
    # ---- Train ----
    model.train()
    total_train_loss = 0.0

    for step, batch in enumerate(train_loader):
        outputs = model(**batch)
        loss = outputs.loss
        total_train_loss += loss.item()

        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    # ---- Evaluate ----
    model.eval()
    total_eval_loss = 0.0
    with torch.no_grad():
        for batch in eval_loader:
            outputs = model(**batch)
            total_eval_loss += outputs.loss.item()

    # Reduce across processes if distributed
    total_train_loss = accelerator.gather(torch.tensor(total_train_loss)).mean().item()
    total_eval_loss = accelerator.gather(torch.tensor(total_eval_loss)).mean().item()

    avg_train_loss = total_train_loss / len(train_loader)
    avg_eval_loss = total_eval_loss / len(eval_loader)
    ppl = math.exp(avg_eval_loss) if avg_eval_loss < 20 else float("inf")

    accelerator.print(
        f"Epoch {epoch+1}/{NUM_EPOCHS} | "
        f"train_loss={avg_train_loss:.4f} | "
        f"eval_loss={avg_eval_loss:.4f} | "
        f"perplexity={ppl:.2f}"
    )

    # Save best
    if avg_eval_loss < best_eval_loss:
        best_eval_loss = avg_eval_loss
        accelerator.wait_for_everyone()
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(output_dir, save_function=accelerator.save)
        tokenizer.save_pretrained(output_dir)
        if PUSH_TO_HUB and accelerator.is_main_process:
            repo.push_to_hub(commit_message=f"epoch {epoch+1} (best so far)")

# Final save
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    accelerator.print("Training complete. Saving final checkpoint…")
unwrapped = accelerator.unwrap_model(model)
unwrapped.save_pretrained(output_dir, save_function=accelerator.save)
tokenizer.save_pretrained(output_dir)
if PUSH_TO_HUB and accelerator.is_main_process:
    repo.push_to_hub(commit_message="final")
```

### Notes / knobs you can tweak

* **Dataset**: switch `load_dataset("wikitext", "wikitext-2-raw-v1")` to your corpus.
* **Whole word masking**: replace `DataCollatorForLanguageModeling` with `DataCollatorForWholeWordMask` if your tokenizer supports it.
* **Longer sequences**: raise `BLOCK_SIZE` (e.g., 256/512) if you have GPU memory.
* **Warmup**: tune `WARMUP_RATIO` (e.g., 0.06–0.1) for smoother training.
* **Hub**: set `PUSH_TO_HUB=True` (and `huggingface-cli login`) to upload checkpoints.

This gives you the exact workflow you asked for: random masking during **training**, **fixed masking** during **evaluation** for stable perplexity, with a robust Accelerate training loop.

