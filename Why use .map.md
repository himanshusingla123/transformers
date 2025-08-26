**Tokenizing an entire dataset efficiently**.

---

### 1. What is the problem?

When we have one pair of sentences, we can directly call:

```python
tokenizer(sentence1, sentence2, truncation=True, padding=True)
```

✅ This works, but...
❌ It tries to tokenize **all sentences at once** and store them in RAM → not good if dataset is large.
❌ Returns just a dictionary of lists, **not a dataset object** → we lose benefits like lazy loading and memory efficiency.

---

### 2. What is the solution?

Instead of tokenizing everything at once, we use the **`.map()`** method of a dataset.

👉 `.map()` means:
“Go through each example in my dataset (or a batch of examples) and apply a function to it, then return a new dataset with the transformed results.”

---

### 3. Defining a tokenization function

```python
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
```

* `example` is **one row** (or a batch of rows if we use `batched=True`) from the dataset.
  Example:

  ```python
  example = {"sentence1": "The sky is blue.", "sentence2": "The ocean is deep."}
  ```
* `tokenizer(...)` converts those sentences into `input_ids`, `attention_mask`, and `token_type_ids`.
* That dictionary is returned and added back into the dataset.

---

### 4. Applying it to the whole dataset

```python
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
```

* `raw_datasets` is your dataset (like from `load_dataset()`).
* `map(tokenize_function, batched=True)`
  🔹 applies `tokenize_function` to the dataset.
  🔹 `batched=True` → instead of processing **one row at a time**, the function gets a **batch of rows** (lists of sentences).
  This is **much faster** because Hugging Face tokenizer (written in Rust) can handle big batches efficiently.

---

### 5. Why not pad here?

They did **not use padding** (`padding=True`) inside `tokenize_function`.
Reason:

* If you pad now, every sentence will be padded to the **maximum length in the entire dataset**. Wasteful!
* Better: pad later at the **batch creation step** (just before feeding into the model).
  That way, you only pad to the **longest sentence in that batch**, which saves memory and time.

---

✅ So in short:

* **Direct tokenization** → works, but not memory-efficient.
* **Dataset.map() with batched=True** → applies tokenizer to whole dataset, keeps it as a dataset object, efficient + flexible.
* **Padding later** → avoids wasting computation.

---