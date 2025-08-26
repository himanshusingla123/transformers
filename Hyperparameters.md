# Complete Guide to Transformers Library Parameters

## 1. Model Loading Parameters

### `torch_dtype`
**Purpose**: Controls the precision/data type of model weights
**Values**: `torch.float32`, `torch.float16`, `torch.bfloat16`, `"auto"`
**Default**: Usually `float32`

**Effects**:
- **float32**: Highest precision, more memory usage (~4GB for 1B params)
- **float16**: Half precision, 50% memory reduction, potential precision loss
- **bfloat16**: Better numerical stability than float16, similar memory savings
- **"auto"**: Automatically chooses based on model configuration

**Example**:
```python
from transformers import AutoModel
import torch

# High precision, more memory
model = AutoModel.from_pretrained("bert-base-uncased", torch_dtype=torch.float32)

# Half precision, less memory
model = AutoModel.from_pretrained("bert-base-uncased", torch_dtype=torch.float16)
```

### `device_map`
**Purpose**: Controls how model layers are distributed across devices
**Values**: `"auto"`, `"balanced"`, `"balanced_low_0"`, `"sequential"`, custom dict
**Default**: `None` (single device)

**Effects**:
- **"auto"**: Automatically distributes layers across available GPUs
- **"balanced"**: Evenly distributes memory usage across devices
- **"sequential"**: Places layers sequentially across devices
- **Custom dict**: Manual control over layer placement

**Example**:
```python
# Automatic distribution
model = AutoModel.from_pretrained("facebook/opt-6.7b", device_map="auto")

# Custom distribution
device_map = {
    "transformer.wte": 0,
    "transformer.h.0": 0,
    "transformer.h.1": 1,
    "lm_head": 1
}
model = AutoModel.from_pretrained("gpt2", device_map=device_map)
```

### `low_cpu_mem_usage`
**Purpose**: Reduces CPU memory usage during model loading
**Values**: `True`, `False`
**Default**: `False`

**Effects**:
- **True**: Loads model with minimal CPU memory footprint
- **False**: Standard loading, may use more CPU memory temporarily

## 2. Generation Parameters

### `max_length` / `max_new_tokens`
**Purpose**: Controls the maximum length of generated text
**Values**: Any positive integer
**Default**: Model-specific (usually 20-1024)

**Effects**:
- **Low values (10-50)**: Short, concise responses
- **Medium values (100-500)**: Balanced responses
- **High values (1000+)**: Long, detailed responses, higher computational cost

**Example**:
```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

# Short generation
output = generator("Hello world", max_new_tokens=10)
# Output: "Hello world, I'm a little bit of a"

# Long generation
output = generator("Hello world", max_new_tokens=100)
# Output: Much longer text...
```

### `temperature`
**Purpose**: Controls randomness in text generation
**Values**: 0.0 to 2.0+ (float)
**Default**: 1.0

**Effects**:
- **0.0-0.3**: Very deterministic, repetitive, safe outputs
- **0.7-1.0**: Balanced creativity and coherence
- **1.5-2.0**: Highly creative, potentially incoherent

**Mathematical Effect**: Softmax temperature scaling
```
P(token) = exp(logit/temperature) / sum(exp(logits/temperature))
```

**Example**:
```python
# Conservative generation
output = generator("The future of AI", temperature=0.1)
# Result: Predictable, focused text

# Creative generation
output = generator("The future of AI", temperature=1.5)
# Result: More diverse, potentially surprising text
```

### `top_p` (Nucleus Sampling)
**Purpose**: Controls diversity by considering only top tokens with cumulative probability p
**Values**: 0.0 to 1.0 (float)
**Default**: 1.0 (disabled)

**Effects**:
- **0.1-0.3**: Very focused, high-probability tokens only
- **0.5-0.7**: Balanced selection
- **0.9-0.95**: Diverse but still coherent
- **1.0**: All tokens considered (disabled)

**Example**:
```python
# Focused generation
output = generator("Science is", top_p=0.2)
# Result: Uses only the most likely words

# Diverse generation
output = generator("Science is", top_p=0.9)
# Result: More varied vocabulary choices
```

### `top_k`
**Purpose**: Limits selection to top-k most likely tokens
**Values**: 1 to vocabulary size (integer)
**Default**: 50

**Effects**:
- **1**: Greedy decoding (most deterministic)
- **10-20**: Focused but some variety
- **50-100**: Standard diversity
- **500+**: High diversity

**Example**:
```python
# Very focused
output = generator("The cat", top_k=5)

# More diverse
output = generator("The cat", top_k=100)
```

### `repetition_penalty`
**Purpose**: Penalizes repeated tokens/phrases
**Values**: 0.0 to 2.0+ (float)
**Default**: 1.0 (no penalty)

**Effects**:
- **1.0**: No repetition penalty
- **1.1-1.2**: Mild penalty, reduces some repetition
- **1.5+**: Strong penalty, may hurt coherence

**Example**:
```python
# No penalty (may repeat)
output = generator("I love", repetition_penalty=1.0)

# With penalty (less repetitive)
output = generator("I love", repetition_penalty=1.2)
```

### `do_sample`
**Purpose**: Enables/disables sampling vs greedy decoding
**Values**: `True`, `False`
**Default**: `False`

**Effects**:
- **False**: Greedy decoding (deterministic, always picks highest probability)
- **True**: Enables sampling (uses temperature, top_p, top_k)

### `num_beams`
**Purpose**: Number of beams for beam search
**Values**: 1 to 20+ (integer)
**Default**: 1 (greedy/sampling)

**Effects**:
- **1**: Greedy or sampling
- **2-5**: Better quality, moderate computational cost
- **10+**: Higher quality, much slower

**Example**:
```python
# Greedy search
output = generator("Hello", num_beams=1)

# Beam search
output = generator("Hello", num_beams=5)
```

## 3. Training Parameters

### `learning_rate`
**Purpose**: Controls step size in gradient descent
**Values**: 1e-6 to 1e-2 (float)
**Default**: 5e-5 for most models

**Effects**:
- **Too low (1e-6)**: Very slow convergence
- **Optimal (1e-5 to 5e-5)**: Good convergence for most tasks
- **Too high (1e-3+)**: May overshoot, unstable training

**Example**:
```python
from transformers import TrainingArguments

# Conservative learning rate
training_args = TrainingArguments(
    learning_rate=1e-5,
    # other parameters...
)

# Aggressive learning rate
training_args = TrainingArguments(
    learning_rate=5e-4,
    # other parameters...
)
```

### `batch_size` (per_device_train_batch_size)
**Purpose**: Number of samples processed together
**Values**: 1 to 1000+ (limited by memory)
**Default**: 8

**Effects**:
- **Small (1-4)**: Less memory, noisier gradients, slower training
- **Medium (8-32)**: Balanced efficiency and stability
- **Large (64+)**: More memory, stabler gradients, faster training

**Memory Usage**: Roughly linear relationship
```
Memory ≈ batch_size × sequence_length × model_size × 4 bytes
```

### `gradient_accumulation_steps`
**Purpose**: Accumulates gradients over multiple mini-batches
**Values**: 1 to 100+ (integer)
**Default**: 1

**Effective batch size = batch_size × gradient_accumulation_steps**

**Effects**:
- **1**: No accumulation
- **4-8**: Simulates larger batch sizes without memory increase
- **16+**: Very large effective batch sizes

### `warmup_steps` / `warmup_ratio`
**Purpose**: Gradually increases learning rate at training start
**Values**: 
- `warmup_steps`: 0 to 1000+ (integer)
- `warmup_ratio`: 0.0 to 0.3 (float)

**Effects**:
- **0**: No warmup, may cause instability
- **100-500 steps** or **0.06-0.1 ratio**: Standard warmup
- **High values**: Slower initial training

### `weight_decay`
**Purpose**: L2 regularization to prevent overfitting
**Values**: 0.0 to 0.1 (float)
**Default**: 0.01

**Effects**:
- **0.0**: No regularization, may overfit
- **0.01**: Standard regularization
- **0.1+**: Strong regularization, may underfit

### `max_grad_norm`
**Purpose**: Clips gradients to prevent exploding gradients
**Values**: 0.1 to 10.0 (float)
**Default**: 1.0

**Effects**:
- **Small (0.1-0.5)**: Very conservative, may slow learning
- **Standard (1.0)**: Good default
- **Large (5.0+)**: Less clipping, risk of instability

## 4. Attention Mechanism Parameters

### `attention_dropout`
**Purpose**: Dropout rate in attention layers
**Values**: 0.0 to 0.5 (float)
**Default**: 0.1

**Effects**:
- **0.0**: No dropout, may overfit
- **0.1**: Standard dropout
- **0.3+**: Strong regularization

### `num_attention_heads`
**Purpose**: Number of attention heads in multi-head attention
**Values**: Model architecture dependent (usually 8, 12, 16)
**Fixed at architecture design**

**Effects**:
- **Fewer heads**: Less parallel attention patterns
- **More heads**: More diverse attention patterns, higher computation

### `hidden_size`
**Purpose**: Dimensionality of model representations
**Values**: Usually 768, 1024, 1536, 2048, 4096
**Fixed at architecture design**

**Effects**:
- **Smaller**: Less capacity, faster, less memory
- **Larger**: More capacity, slower, more memory

## 5. Tokenization Parameters

### `max_length` (tokenization)
**Purpose**: Maximum sequence length for tokenization
**Values**: 1 to model's max_position_embeddings
**Default**: 512 for many models

**Effects**:
- **Short (128)**: Faster processing, may truncate important info
- **Long (1024+)**: Slower processing, captures more context

### `truncation`
**Purpose**: How to handle sequences longer than max_length
**Values**: `True`, `False`, `"longest_first"`, `"only_first"`, `"only_second"`
**Default**: `False`

### `padding`
**Purpose**: How to pad sequences to equal length
**Values**: `True`, `False`, `"longest"`, `"max_length"`
**Default**: `False`

**Example**:
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Short sequences, no padding
tokens = tokenizer("Hello world", max_length=512, padding=False)

# Pad to max length
tokens = tokenizer("Hello world", max_length=512, padding="max_length")
```

## 6. Model-Specific Parameters

### BERT Parameters

#### `hidden_dropout_prob`
**Values**: 0.0 to 0.5
**Default**: 0.1
**Purpose**: Dropout in hidden layers

#### `attention_probs_dropout_prob`
**Values**: 0.0 to 0.5
**Default**: 0.1
**Purpose**: Dropout in attention probabilities

### GPT Parameters

#### `n_layer`
**Purpose**: Number of transformer layers
**Values**: 6, 12, 24, 36, 48+ (architecture dependent)

#### `n_head`
**Purpose**: Number of attention heads per layer
**Values**: 8, 12, 16, 20+ (architecture dependent)

### T5 Parameters

#### `d_model`
**Purpose**: Model dimensionality
**Values**: 512, 768, 1024 (architecture dependent)

#### `d_ff`
**Purpose**: Feed-forward layer dimensionality
**Values**: Usually 4 × d_model

## 7. Performance and Memory Parameters

### `use_cache`
**Purpose**: Cache key-value pairs for faster generation
**Values**: `True`, `False`
**Default**: `True`

**Effects**:
- **True**: Faster generation, more memory usage
- **False**: Slower generation, less memory usage

### `output_attentions`
**Purpose**: Return attention weights
**Values**: `True`, `False`
**Default**: `False`

**Memory Impact**: Significant memory increase when `True`

### `output_hidden_states`
**Purpose**: Return all hidden states
**Values**: `True`, `False`
**Default**: `False`

**Memory Impact**: Large memory increase when `True`

## 8. Practical Parameter Combinations

### Conservative Generation (Safe, Predictable)
```python
generation_config = {
    "temperature": 0.1,
    "top_p": 0.9,
    "top_k": 10,
    "repetition_penalty": 1.1,
    "do_sample": True,
    "max_new_tokens": 100
}
```

### Creative Generation (Diverse, Interesting)
```python
generation_config = {
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 50,
    "repetition_penalty": 1.2,
    "do_sample": True,
    "max_new_tokens": 200
}
```

### High-Quality Generation (Best Results, Slower)
```python
generation_config = {
    "num_beams": 5,
    "early_stopping": True,
    "no_repeat_ngram_size": 3,
    "max_new_tokens": 150
}
```

### Memory-Efficient Training
```python
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size = 16
    fp16=True,  # Half precision
    dataloader_pin_memory=False,
    learning_rate=2e-5,
    warmup_ratio=0.1
)
```

### Fast Training (Less Accuracy)
```python
training_args = TrainingArguments(
    per_device_train_batch_size=32,
    learning_rate=5e-4,  # Higher learning rate
    num_train_epochs=3,  # Fewer epochs
    warmup_ratio=0.06,
    save_strategy="epoch"
)
```

## 9. Parameter Interaction Effects

### Temperature + Top-p Interaction
- **High temp + Low top_p**: Diverse selection from restricted vocabulary
- **Low temp + High top_p**: Conservative selection from full vocabulary
- **High temp + High top_p**: Maximum diversity and creativity
- **Low temp + Low top_p**: Very focused and deterministic

### Batch Size + Learning Rate
- **Large batch + High LR**: Risk of overshooting
- **Small batch + Low LR**: Very slow convergence
- **Optimal**: Scale learning rate with batch size (linear or sqrt scaling)

### Sequence Length + Batch Size
Memory usage scales quadratically: `O(seq_len² × batch_size)`

## 10. Common Parameter Mistakes

### 1. Memory Issues
```python
# BAD: Will likely cause OOM
model = AutoModel.from_pretrained("large-model", torch_dtype=torch.float32)

# GOOD: Use half precision for large models
model = AutoModel.from_pretrained("large-model", torch_dtype=torch.float16)
```

### 2. Generation Quality Issues
```python
# BAD: May produce repetitive text
generator("Hello", temperature=0.0, repetition_penalty=1.0)

# GOOD: Balanced parameters
generator("Hello", temperature=0.7, repetition_penalty=1.1, top_p=0.9)
```

### 3. Training Instability
```python
# BAD: High learning rate without warmup
TrainingArguments(learning_rate=1e-3, warmup_steps=0)

# GOOD: Appropriate LR with warmup
TrainingArguments(learning_rate=2e-5, warmup_ratio=0.1)
```

This guide covers the most important parameters in the Transformers library. The optimal values depend on your specific task, model size, hardware constraints, and quality requirements. Always start with default values and adjust gradually based on your results.