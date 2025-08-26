# Comprehensive Transformers Library Methods & Classes

## Core Classes & Methods Table

| Category | Class/Method | Purpose | Key Parameters | Usage Example |
|----------|-------------|---------|----------------|---------------|
| **Loading & Tokenization** | | | | |
| | `AutoTokenizer.from_pretrained()` | Load tokenizer automatically | `model_name`, `cache_dir`, `use_fast` | `tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")` |
| | `AutoModel.from_pretrained()` | Load model automatically | `model_name`, `config`, `cache_dir` | `model = AutoModel.from_pretrained("bert-base-uncased")` |
| | `AutoConfig.from_pretrained()` | Load model configuration | `model_name`, `num_labels`, `dropout` | `config = AutoConfig.from_pretrained("bert-base-uncased")` |
| | `tokenizer.encode()` | Convert text to token IDs | `text`, `max_length`, `truncation`, `padding` | `tokens = tokenizer.encode("Hello world", max_length=512)` |
| | `tokenizer.decode()` | Convert token IDs back to text | `token_ids`, `skip_special_tokens` | `text = tokenizer.decode([101, 7592, 2088, 102])` |
| | `tokenizer()` | Tokenize with all options | `text`, `return_tensors`, `padding`, `truncation` | `inputs = tokenizer("Hello", return_tensors="pt")` |
| **Model Classes** | | | | |
| | `AutoModelForSequenceClassification` | Text classification models | `num_labels`, `config` | `model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)` |
| | `AutoModelForTokenClassification` | Token-level classification (NER) | `num_labels`, `config` | `model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=9)` |
| | `AutoModelForQuestionAnswering` | Question answering models | `config` | `model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")` |
| | `AutoModelForMaskedLM` | Masked language modeling | `config` | `model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")` |
| | `AutoModelForCausalLM` | Causal language modeling (GPT-style) | `config` | `model = AutoModelForCausalLM.from_pretrained("gpt2")` |
| | `AutoModelForSeq2SeqLM` | Sequence-to-sequence models | `config` | `model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")` |
| **Training** | | | | |
| | `TrainingArguments` | Training configuration | `output_dir`, `learning_rate`, `num_train_epochs` | `args = TrainingArguments(output_dir="./results", learning_rate=5e-5)` |
| | `Trainer` | Main training orchestrator | `model`, `args`, `train_dataset`, `eval_dataset` | `trainer = Trainer(model=model, args=training_args, train_dataset=train_ds)` |
| | `trainer.train()` | Execute training | `resume_from_checkpoint` | `trainer.train()` |
| | `trainer.evaluate()` | Evaluate model | `eval_dataset`, `metric_key_prefix` | `results = trainer.evaluate()` |
| | `trainer.predict()` | Make predictions | `test_dataset` | `predictions = trainer.predict(test_dataset)` |
| | `trainer.save_model()` | Save trained model | `output_dir` | `trainer.save_model("./my_model")` |
| | `trainer.push_to_hub()` | Upload model to Hub | `repo_name`, `commit_message` | `trainer.push_to_hub("my-finetuned-model")` |
| **Data Handling** | | | | |
| | `DataCollatorWithPadding` | Batch data with padding | `tokenizer`, `padding`, `max_length` | `collator = DataCollatorWithPadding(tokenizer=tokenizer)` |
| | `DataCollatorForLanguageModeling` | MLM data collation | `tokenizer`, `mlm_probability` | `collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)` |
| | `DataCollatorForSeq2Seq` | Seq2seq data collation | `tokenizer`, `model`, `padding` | `collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)` |
| **Pipelines** | | | | |
| | `pipeline()` | Create inference pipeline | `task`, `model`, `tokenizer` | `classifier = pipeline("sentiment-analysis")` |
| | `pipeline("text-classification")` | Text classification pipeline | `model`, `tokenizer`, `device` | `classifier = pipeline("text-classification", model="bert-base-uncased")` |
| | `pipeline("token-classification")` | NER pipeline | `model`, `aggregation_strategy` | `ner = pipeline("token-classification", model="dbmdz/bert-large-cased-finetuned-conll03-english")` |
| | `pipeline("question-answering")` | QA pipeline | `model`, `tokenizer` | `qa = pipeline("question-answering")` |
| | `pipeline("text-generation")` | Text generation pipeline | `model`, `max_length`, `temperature` | `generator = pipeline("text-generation", model="gpt2")` |
| **Model Methods** | | | | |
| | `model()` | Forward pass | `input_ids`, `attention_mask`, `labels` | `outputs = model(input_ids=inputs["input_ids"])` |
| | `model.forward()` | Explicit forward pass | `input_ids`, `attention_mask`, `token_type_ids` | `outputs = model.forward(**inputs)` |
| | `model.generate()` | Generate text (for generative models) | `input_ids`, `max_length`, `temperature`, `do_sample` | `generated = model.generate(input_ids, max_length=50)` |
| | `model.save_pretrained()` | Save model and config | `save_directory` | `model.save_pretrained("./my_model")` |
| | `model.push_to_hub()` | Upload to Hub | `repo_name`, `commit_message` | `model.push_to_hub("my-model")` |
| | `model.eval()` | Set to evaluation mode | None | `model.eval()` |
| | `model.train()` | Set to training mode | None | `model.train()` |
| **Configuration** | | | | |
| | `config.save_pretrained()` | Save configuration | `save_directory` | `config.save_pretrained("./my_config")` |
| | `config.push_to_hub()` | Upload config to Hub | `repo_name` | `config.push_to_hub("my-config")` |
| **Callbacks** | | | | |
| | `EarlyStoppingCallback` | Early stopping during training | `early_stopping_patience`, `early_stopping_threshold` | `callback = EarlyStoppingCallback(early_stopping_patience=3)` |
| | `TensorBoardCallback` | TensorBoard logging | `tb_writer` | `callback = TensorBoardCallback()` |
| | `WandbCallback` | Weights & Biases integration | `project`, `name` | `callback = WandbCallback()` |
| **Utilities** | | | | |
| | `logging.set_verbosity_info()` | Set logging level | None | `transformers.logging.set_verbosity_info()` |
| | `set_seed()` | Set random seed | `seed` | `transformers.set_seed(42)` |
| | `is_torch_available()` | Check PyTorch availability | None | `transformers.is_torch_available()` |

## Detailed Explanations of Complex Methods

### 1. AutoTokenizer Methods
**Purpose**: Handle text preprocessing and tokenization
- `from_pretrained()`: Downloads and caches tokenizer
- `encode()`: Basic tokenization to IDs
- `__call__()`: Full tokenization with all options
- `batch_encode_plus()`: Batch processing
- `pad()`: Add padding to sequences
- `truncate_sequences()`: Handle long sequences

### 2. Model Forward Pass
**Purpose**: Process inputs through the model
```python
outputs = model(
    input_ids=input_ids,           # Token IDs
    attention_mask=attention_mask, # Padding mask
    token_type_ids=token_type_ids, # Segment IDs (BERT)
    labels=labels,                 # For loss calculation
    output_attentions=True,        # Return attention weights
    output_hidden_states=True,     # Return all layer outputs
)
```

### 3. Training Arguments Parameters
**Purpose**: Configure all aspects of training
- **Learning Rate**: `learning_rate`, `lr_scheduler_type`
- **Batching**: `per_device_train_batch_size`, `gradient_accumulation_steps`
- **Evaluation**: `eval_strategy`, `eval_steps`, `load_best_model_at_end`
- **Saving**: `save_strategy`, `save_steps`, `save_total_limit`
- **Optimization**: `warmup_steps`, `weight_decay`, `adam_epsilon`
- **Mixed Precision**: `fp16`, `bf16`

### 4. Pipeline Advanced Usage
**Purpose**: Ready-to-use inference interfaces
```python
# Text Classification
classifier = pipeline("text-classification", 
                     model="cardiffnlp/twitter-roberta-base-sentiment",
                     return_all_scores=True)

# Token Classification with aggregation
ner = pipeline("token-classification",
               model="dbmdz/bert-large-cased-finetuned-conll03-english",
               aggregation_strategy="simple")

# Text Generation with parameters
generator = pipeline("text-generation",
                    model="gpt2",
                    max_length=50,
                    temperature=0.7,
                    do_sample=True)
```

### 5. Data Collator Types
**Purpose**: Handle batching and padding for different tasks
- **WithPadding**: Standard padding for classification
- **ForLanguageModeling**: MLM with masking
- **ForSeq2Seq**: Encoder-decoder tasks
- **ForWholeWordMask**: Whole word masking
- **ForPermutationLanguageModeling**: XLNet-style

### 6. Model Generation Methods
**Purpose**: Generate text with various strategies
```python
generated = model.generate(
    input_ids,
    max_length=50,        # Maximum sequence length
    min_length=10,        # Minimum sequence length
    temperature=0.8,      # Sampling temperature
    top_k=50,            # Top-k sampling
    top_p=0.95,          # Nucleus sampling
    do_sample=True,       # Enable sampling
    num_beams=5,         # Beam search
    early_stopping=True,  # Stop when EOS
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
```

### 7. Custom Callbacks
**Purpose**: Extend training behavior
```python
class CustomCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        # Training start logic
        pass
    
    def on_epoch_end(self, args, state, control, **kwargs):
        # End of epoch logic
        pass
    
    def on_evaluate(self, args, state, control, **kwargs):
        # During evaluation
        pass
```

### 8. Hub Integration Methods
**Purpose**: Upload and share models
- `push_to_hub()`: Upload model/tokenizer
- `create_repo()`: Create repository
- `upload_file()`: Upload individual files
- `delete_repo()`: Delete repository

### 9. Model Output Objects
**Purpose**: Structured outputs from models
- `BaseModelOutput`: Basic model outputs
- `SequenceClassifierOutput`: Classification outputs
- `TokenClassifierOutput`: Token classification outputs
- `QuestionAnsweringModelOutput`: QA model outputs
- `Seq2SeqModelOutput`: Sequence-to-sequence outputs

Each output contains:
- `last_hidden_state`: Final layer representations
- `hidden_states`: All layer representations (if requested)
- `attentions`: Attention weights (if requested)
- `logits`: Raw model predictions
- `loss`: Computed loss (if labels provided)

This comprehensive table covers all major methods and classes in the Transformers library, from basic loading to advanced training and inference capabilities.