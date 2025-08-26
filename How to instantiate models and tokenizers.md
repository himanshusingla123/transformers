# Complete Guide to Model Instantiation in Transformers

## Table of Contents
1. [Direct Architecture Instantiation](#1-direct-architecture-instantiation)
2. [Auto Classes (Recommended)](#2-auto-classes-recommended)
3. [Framework-Specific Instantiation](#3-framework-specific-instantiation)
4. [All Model Loading Cases](#4-all-model-loading-cases)
5. [Tokenizer Instantiation Methods](#5-tokenizer-instantiation-methods)
6. [Configuration Loading](#6-configuration-loading)
7. [Custom and Local Models](#7-custom-and-local-models)
8. [Model Card Analysis](#8-model-card-analysis)
9. [Best Practices and Recommendations](#9-best-practices-and-recommendations)

## 1. Direct Architecture Instantiation

### What it means
Direct instantiation involves importing and using specific model classes that are tied to particular architectures (BERT, GPT-2, T5, etc.).

### Syntax Pattern
```python
from transformers import [ArchitectureName][Task], [ArchitectureName]Tokenizer
```

### Complete Examples

#### BERT Family
```python
# BERT Base
from transformers import BertTokenizer, BertForMaskedLM
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

# BERT for Sequence Classification
from transformers import BertTokenizer, BertForSequenceClassification
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# RoBERTa (BERT variant)
from transformers import RobertaTokenizer, RobertaForMaskedLM
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForMaskedLM.from_pretrained("roberta-base")

# DistilBERT (compressed BERT)
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
```

#### GPT Family
```python
# GPT-2
from transformers import GPT2Tokenizer, GPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# GPT-2 for Sequence Classification
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2ForSequenceClassification.from_pretrained("gpt2")
```

#### T5 Family
```python
# T5 Base
from transformers import T5Tokenizer, T5ForConditionalGeneration
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# T5 for Question Answering
from transformers import T5Tokenizer, T5ForQuestionAnswering
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForQuestionAnswering.from_pretrained("t5-base")
```

#### CamemBERT (French BERT)
```python
from transformers import CamembertTokenizer, CamembertForMaskedLM
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
model = CamembertForMaskedLM.from_pretrained("camembert-base")

# For other tasks
from transformers import CamembertForSequenceClassification
model = CamembertForSequenceClassification.from_pretrained("camembert-base")
```

### Advantages of Direct Instantiation
1. **Explicit Control**: You know exactly which architecture you're using
2. **Type Safety**: IDEs provide better autocomplete and type checking
3. **Architecture-Specific Features**: Access to model-specific methods and attributes
4. **Documentation**: Clear documentation for each specific class

### Disadvantages of Direct Instantiation
1. **Architecture Lock-in**: Code is tied to specific architecture
2. **Less Flexible**: Harder to switch between models
3. **More Code Changes**: Requires imports changes when switching models
4. **Maintenance Overhead**: Need to update imports for different experiments

## 2. Auto Classes (Recommended)

### What Auto Classes Do
Auto classes automatically detect the correct architecture from the model checkpoint and instantiate the appropriate class.

### Auto Class Types

#### For Models
```python
from transformers import (
    AutoModel,              # Base model (encoder only)
    AutoModelForMaskedLM,   # Masked language modeling
    AutoModelForCausalLM,   # Causal language modeling
    AutoModelForSeq2SeqLM,  # Sequence-to-sequence
    AutoModelForSequenceClassification,  # Classification
    AutoModelForQuestionAnswering,       # Question answering
    AutoModelForTokenClassification,     # Token classification (NER)
    AutoModelForMultipleChoice,          # Multiple choice
    AutoModelForNextSentencePrediction,  # Next sentence prediction
)
```

#### For Tokenizers
```python
from transformers import AutoTokenizer
```

#### For Configurations
```python
from transformers import AutoConfig
```

### Complete Auto Class Examples

#### Basic Usage
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Works with any masked LM model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

# Same code works with RoBERTa
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForMaskedLM.from_pretrained("roberta-base")

# Same code works with CamemBERT
tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = AutoModelForMaskedLM.from_pretrained("camembert-base")
```

#### Task-Specific Auto Classes
```python
# Text Classification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Question Answering
from transformers import AutoModelForQuestionAnswering
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")

# Text Generation
from transformers import AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Summarization/Translation
from transformers import AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
```

### Advantages of Auto Classes
1. **Architecture Agnostic**: Same code works with different models
2. **Easy Model Switching**: Change only the model name
3. **Future-Proof**: Automatically supports new architectures
4. **Cleaner Code**: Less import statements
5. **Experimentation Friendly**: Easy to test different models

### Disadvantages of Auto Classes
1. **Less Explicit**: Not immediately clear which architecture is loaded
2. **Potential Issues**: May auto-detect incorrectly in rare cases
3. **Less Architecture-Specific Control**: Some model-specific features might be less accessible

## 3. Framework-Specific Instantiation

### PyTorch (Default)
```python
# Direct instantiation
from transformers import BertModel, BertTokenizer
model = BertModel.from_pretrained("bert-base-uncased")

# Auto classes
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("bert-base-uncased")
```

### TensorFlow
```python
# Direct instantiation
from transformers import TFBertModel, BertTokenizer
model = TFBertModel.from_pretrained("bert-base-uncased")

# Auto classes
from transformers import TFAutoModel, AutoTokenizer
model = TFAutoModel.from_pretrained("bert-base-uncased")

# Task-specific TF models
from transformers import TFAutoModelForSequenceClassification
model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
```

### JAX/Flax
```python
# Direct instantiation
from transformers import FlaxBertModel, BertTokenizer
model = FlaxBertModel.from_pretrained("bert-base-uncased")

# Auto classes
from transformers import FlaxAutoModel, AutoTokenizer
model = FlaxAutoModel.from_pretrained("bert-base-uncased")
```

### Framework Prefix Patterns
- **PyTorch**: No prefix (default)
- **TensorFlow**: `TF` prefix (e.g., `TFBertModel`, `TFAutoModel`)
- **JAX/Flax**: `Flax` prefix (e.g., `FlaxBertModel`, `FlaxAutoModel`)

## 4. All Model Loading Cases

### Case 1: From Hugging Face Hub
```python
# Most common case
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
```

### Case 2: From Local Directory
```python
# Model saved locally
tokenizer = AutoTokenizer.from_pretrained("./my-saved-model/")
model = AutoModel.from_pretrained("./my-saved-model/")
```

### Case 3: From URL
```python
# Direct URL to model files
tokenizer = AutoTokenizer.from_pretrained("https://example.com/model/")
model = AutoModel.from_pretrained("https://example.com/model/")
```

### Case 4: With Custom Configuration
```python
from transformers import AutoConfig, AutoModel
config = AutoConfig.from_pretrained("bert-base-uncased")
config.num_labels = 5  # Modify configuration
model = AutoModel.from_pretrained("bert-base-uncased", config=config)
```

### Case 5: Random Initialization
```python
from transformers import AutoConfig, AutoModel
config = AutoConfig.from_pretrained("bert-base-uncased")
model = AutoModel.from_config(config)  # Randomly initialized weights
```

### Case 6: With Additional Parameters
```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "bert-large-uncased",
    torch_dtype=torch.float16,      # Half precision
    device_map="auto",              # Auto device placement
    trust_remote_code=True,         # Allow custom code
    use_auth_token=True,            # Use HF auth token
    revision="main",                # Specific git revision
    cache_dir="./cache",            # Custom cache directory
)
```

### Case 7: Conditional Loading
```python
from transformers import AutoTokenizer, AutoModel
import torch

model_name = "bert-base-uncased"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
```

### Case 8: Loading Specific Model Variants
```python
# Different sizes of same architecture
models = [
    "bert-base-uncased",     # 110M parameters
    "bert-large-uncased",    # 340M parameters
    "distilbert-base-uncased", # 66M parameters (compressed)
]

for model_name in models:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    print(f"Loaded {model_name}: {model.num_parameters()} parameters")
```

## 5. Tokenizer Instantiation Methods

### Auto Tokenizer (Recommended)
```python
from transformers import AutoTokenizer

# Automatically detects tokenizer type
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")    # BertTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")                # GPT2Tokenizer  
tokenizer = AutoTokenizer.from_pretrained("t5-base")             # T5Tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")  # BartTokenizer
```

### Direct Tokenizer Classes
```python
# Specific tokenizer classes
from transformers import (
    BertTokenizer,
    GPT2Tokenizer,
    T5Tokenizer,
    BartTokenizer,
    RobertaTokenizer,
    XLNetTokenizer,
    AlbertTokenizer,
    DistilBertTokenizer,
)

# Usage
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
```

### Tokenizer with Custom Parameters
```python
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased",
    do_lower_case=True,         # Convert to lowercase
    strip_accents=True,         # Remove accents
    clean_text=True,            # Clean text
    tokenize_chinese_chars=True, # Handle Chinese characters
    never_split=["[UNK]"],      # Never split these tokens
)
```

### Fast vs Slow Tokenizers
```python
# Fast tokenizer (Rust-based, recommended)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

# Slow tokenizer (Python-based)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=False)

# Check tokenizer type
print(type(tokenizer))  # BertTokenizerFast or BertTokenizer
```

## 6. Configuration Loading

### Auto Configuration
```python
from transformers import AutoConfig

# Load configuration
config = AutoConfig.from_pretrained("bert-base-uncased")
print(config.hidden_size)      # 768
print(config.num_attention_heads)  # 12
print(config.num_hidden_layers)    # 12
```

### Direct Configuration Classes
```python
from transformers import BertConfig, GPT2Config, T5Config

# Specific config classes
bert_config = BertConfig.from_pretrained("bert-base-uncased")
gpt2_config = GPT2Config.from_pretrained("gpt2")
t5_config = T5Config.from_pretrained("t5-base")
```

### Custom Configuration
```python
from transformers import AutoConfig, AutoModel

# Modify existing configuration
config = AutoConfig.from_pretrained("bert-base-uncased")
config.hidden_dropout_prob = 0.2
config.attention_probs_dropout_prob = 0.2

# Use custom config
model = AutoModel.from_pretrained("bert-base-uncased", config=config)
```

### Configuration for New Models
```python
from transformers import BertConfig, BertModel

# Create completely custom configuration
config = BertConfig(
    vocab_size=30522,
    hidden_size=512,         # Smaller than base
    num_hidden_layers=6,     # Fewer layers
    num_attention_heads=8,   # Fewer heads
    intermediate_size=2048,
)

# Initialize model with custom config
model = BertModel(config)  # Random weights
```

## 7. Custom and Local Models

### Loading from Local Directory
```python
# Directory structure:
# ./my-model/
# ├── config.json
# ├── pytorch_model.bin (or model.safetensors)
# ├── tokenizer.json
# ├── tokenizer_config.json
# └── vocab.txt

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("./my-model/")
model = AutoModel.from_pretrained("./my-model/")
```

### Loading Custom Models from Hub
```python
# Models with custom code
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/DialoGPT-large", 
    trust_remote_code=True  # Required for custom code
)
model = AutoModel.from_pretrained(
    "microsoft/DialoGPT-large",
    trust_remote_code=True
)
```

### Loading from Private Repositories
```python
from transformers import AutoTokenizer, AutoModel

# Using authentication token
tokenizer = AutoTokenizer.from_pretrained(
    "your-username/private-model",
    use_auth_token=True  # Uses stored HF token
)

# Or provide token explicitly
tokenizer = AutoTokenizer.from_pretrained(
    "your-username/private-model",
    use_auth_token="hf_your_token_here"
)
```

### Loading Specific Revisions
```python
# Load specific git revision/branch/tag
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased",
    revision="main"      # or specific commit hash
)

model = AutoModel.from_pretrained(
    "bert-base-uncased",
    revision="v1.0"      # or tag name
)
```

## 8. Model Card Analysis

### What is a Model Card?
A model card is a document that provides essential information about a machine learning model, including:

- **Model Description**: Architecture, size, training data
- **Intended Use**: What tasks the model was designed for
- **Limitations**: What the model cannot do well
- **Bias and Fairness**: Known biases and ethical considerations
- **Training Details**: Datasets, hyperparameters, training procedure
- **Evaluation Results**: Performance metrics on various tasks

### Accessing Model Cards
```python
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
import json

model_name = "bert-base-uncased"

# Load model card (README.md)
try:
    card_path = hf_hub_download(repo_id=model_name, filename="README.md")
    with open(card_path, 'r') as f:
        model_card = f.read()
    print("Model Card Summary:")
    print(model_card[:500])  # First 500 characters
except:
    print("No model card found")
```

### Key Information to Check

#### 1. Training Data
```python
# Example of what to look for in model cards:
training_info = {
    "bert-base-uncased": {
        "data": "BooksCorpus + English Wikipedia",
        "size": "3.3B words",
        "languages": ["English"],
        "date": "Pre-2019"
    },
    "gpt2": {
        "data": "WebText dataset",
        "size": "8M web pages",
        "languages": ["English"],
        "filtering": "Minimal content filtering"
    }
}
```

#### 2. Model Limitations
```python
# Common limitations to be aware of:
limitations = {
    "language_models": [
        "May generate biased or toxic content",
        "Can hallucinate facts",
        "Performance degrades on out-of-distribution data",
        "May memorize training data"
    ],
    "bert_models": [
        "Trained on pre-2019 data",
        "May reflect historical biases",
        "Limited to 512 tokens",
        "Requires fine-tuning for downstream tasks"
    ]
}
```

#### 3. Ethical Considerations
```python
# Questions to ask when using any model:
ethical_checklist = [
    "What biases might exist in the training data?",
    "Is the model appropriate for my use case?",
    "Could the model cause harm if used incorrectly?",
    "Are there better alternatives available?",
    "Do I need additional safety measures?"
]
```

### Programmatic Model Card Access
```python
from huggingface_hub import model_info

def analyze_model_card(model_name):
    """Analyze key information from a model's metadata."""
    try:
        info = model_info(model_name)
        
        analysis = {
            "model_name": model_name,
            "library": info.library_name,
            "task": info.pipeline_tag,
            "languages": getattr(info, 'language', None),
            "license": getattr(info, 'license', None),
            "model_size": len(info.siblings) if info.siblings else "Unknown",
            "downloads": info.downloads,
            "likes": info.likes,
        }
        
        return analysis
    except Exception as e:
        return {"error": str(e)}

# Example usage
models_to_analyze = [
    "bert-base-uncased",
    "gpt2",
    "t5-base",
    "distilbert-base-uncased"
]

for model in models_to_analyze:
    analysis = analyze_model_card(model)
    print(f"\n{model}:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")
```

## 9. Best Practices and Recommendations

### 1. Use Auto Classes
```python
# ✅ RECOMMENDED: Architecture-agnostic
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model_for_classification(model_name, num_labels):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels
    )
    return tokenizer, model

# Easy to switch models
models = ["bert-base-uncased", "roberta-base", "distilbert-base-uncased"]
for model_name in models:
    tokenizer, model = load_model_for_classification(model_name, 2)
    print(f"Loaded {model_name} successfully")
```

### 2. Handle Different Scenarios
```python
import torch
from transformers import AutoTokenizer, AutoModel

def smart_model_loading(model_name, **kwargs):
    """Smart model loading with error handling and optimization."""
    
    # Check available memory and adjust accordingly
    if torch.cuda.is_available():
        device = "cuda"
        # Use half precision for large models to save memory
        if "large" in model_name.lower():
            kwargs.setdefault("torch_dtype", torch.float16)
    else:
        device = "cpu"
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with optimizations
        model = AutoModel.from_pretrained(model_name, **kwargs)
        model = model.to(device)
        
        print(f"✅ Successfully loaded {model_name} on {device}")
        print(f"   Parameters: {model.num_parameters():,}")
        
        return tokenizer, model, device
        
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {e}")
        return None, None, None

# Usage examples
models_to_test = [
    "bert-base-uncased",
    "bert-large-uncased",
    "distilbert-base-uncased"
]

for model_name in models_to_test:
    tokenizer, model, device = smart_model_loading(model_name)
```

### 3. Version and Reproducibility
```python
from transformers import AutoTokenizer, AutoModel
import transformers

# Always log versions for reproducibility
print(f"Transformers version: {transformers.__version__}")

# Pin specific model revisions for reproducible results
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased",
    revision="main"  # or specific commit hash
)
model = AutoModel.from_pretrained(
    "bert-base-uncased", 
    revision="main"
)
```

### 4. Error Handling and Fallbacks
```python
from transformers import AutoTokenizer, AutoModel

def robust_model_loading(primary_model, fallback_models=None):
    """Load model with fallback options."""
    
    if fallback_models is None:
        fallback_models = ["distilbert-base-uncased", "bert-base-uncased"]
    
    models_to_try = [primary_model] + fallback_models
    
    for model_name in models_to_try:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            print(f"✅ Successfully loaded {model_name}")
            return tokenizer, model
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
            continue
    
    raise Exception("Failed to load any model")

# Usage
tokenizer, model = robust_model_loading(
    primary_model="some-custom-model",
    fallback_models=["bert-base-uncased", "distilbert-base-uncased"]
)
```

### 5. Memory and Performance Optimization
```python
import torch
from transformers import AutoTokenizer, AutoModel

def memory_efficient_loading(model_name):
    """Load model with memory optimizations."""
    
    # Configuration for memory efficiency
    loading_kwargs = {
        "torch_dtype": torch.float16,  # Half precision
        "low_cpu_mem_usage": True,     # Reduce CPU memory during loading
        "device_map": "auto",          # Auto device placement for multi-GPU
    }
    
    # Load with optimizations
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, **loading_kwargs)
    
    return tokenizer, model

# For large models
tokenizer, model = memory_efficient_loading("microsoft/DialoGPT-large")
```

### 6. Task-Specific Best Practices

#### Text Classification
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def setup_classification_model(model_name, num_labels, label_names=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label={i: label for i, label in enumerate(label_names)} if label_names else None,
        label2id={label: i for i, label in enumerate(label_names)} if label_names else None
    )
    return tokenizer, model

# Usage
labels = ["positive", "negative", "neutral"]
tokenizer, model = setup_classification_model("bert-base-uncased", 3, labels)
```

#### Text Generation
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

def setup_generation_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer, model

# Usage
tokenizer, model = setup_generation_model("gpt2")
```

### 7. When to Use Direct vs Auto Classes

#### Use Direct Classes When:
- You need architecture-specific features
- Working with a specific model family extensively
- Need maximum type safety and IDE support
- Performance is critical (slight overhead in Auto classes)
- Using model-specific methods regularly

```python
# Example: Using BERT-specific features
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Access BERT-specific methods
embeddings = model.embeddings.word_embeddings.weight
attention_weights = model.encoder.layer[0].attention.self
```

#### Use Auto Classes When:
- Prototyping and experimentation
- Building architecture-agnostic pipelines
- Comparing different models
- Building production systems that might use different models
- Following general best practices

```python
# Example: Architecture-agnostic pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def build_classifier_pipeline(model_name, num_labels):
    """Works with any classification model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels
    )
    return tokenizer, model

# Easy to test different architectures
models = ["bert-base-uncased", "roberta-base", "distilbert-base-uncased"]
results = {}

for model_name in models:
    tokenizer, model = build_classifier_pipeline(model_name, 2)
    # ... run evaluation ...
    results[model_name] = evaluation_score
```

This comprehensive guide covers all the major ways to instantiate models and tokenizers in the Transformers library, along with best practices, error handling, and optimization strategies for different use cases.