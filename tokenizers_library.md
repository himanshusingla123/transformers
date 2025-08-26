# Complete Hugging Face Tokenizers Library Methods Reference

This comprehensive table covers all methods available in the Hugging Face `tokenizers` library, organized by class and category.

## Core Tokenizer Class Methods

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | `model, normalizer=None, pre_tokenizer=None, post_processor=None, decoder=None` | Initialize tokenizer with components | Tokenizer |
| `__call__()` | Core | `text, text_pair=None, add_special_tokens=True, padding=False, truncation=False, max_length=None, stride=0, pad_to_multiple_of=None, return_tensors=None, return_token_type_ids=None, return_attention_mask=None, return_overflowing_tokens=False, return_special_tokens_mask=False, return_offsets_mapping=False, return_length=False, verbose=True` | Main encoding method | Encoding or List[Encoding] |
| `encode()` | Core | `sequence, pair=None, is_pretokenized=False, add_special_tokens=True` | Encode a single sequence | Encoding |
| `encode_batch()` | Core | `input, is_pretokenized=False, add_special_tokens=True` | Encode multiple sequences | List[Encoding] |
| `decode()` | Core | `ids, skip_special_tokens=True` | Decode token IDs to string | str |
| `decode_batch()` | Core | `sequences, skip_special_tokens=True` | Decode multiple sequences | List[str] |
| `token_to_id()` | Vocab | `token` | Get token ID for token string | int or None |
| `id_to_token()` | Vocab | `id` | Get token string for token ID | str or None |
| `get_vocab()` | Vocab | `with_added_tokens=True` | Get complete vocabulary | Dict[str, int] |
| `get_vocab_size()` | Vocab | `with_added_tokens=True` | Get vocabulary size | int |
| `enable_truncation()` | Config | `max_length, stride=0, strategy='longest_first', direction='right'` | Enable truncation with parameters | None |
| `disable_truncation()` | Config | None | Disable truncation | None |
| `enable_padding()` | Config | `direction='right', pad_id=0, pad_type_id=0, pad_token='[PAD]', length=None, pad_to_multiple_of=None` | Enable padding with parameters | None |
| `disable_padding()` | Config | None | Disable padding | None |
| `add_tokens()` | Vocab | `tokens` | Add new tokens to vocabulary | int |
| `add_special_tokens()` | Vocab | `tokens` | Add new special tokens | int |
| `train()` | Training | `files, trainer=None` | Train tokenizer on files | None |
| `train_from_iterator()` | Training | `iterator, trainer=None, length=None` | Train from iterator | None |
| `save()` | IO | `path, pretty=True` | Save tokenizer to file | None |
| `from_file()` | IO | `path` | Load tokenizer from file | Tokenizer |
| `from_str()` | IO | `json_str` | Load tokenizer from JSON string | Tokenizer |
| `from_pretrained()` | IO | `identifier, revision='main', auth_token=None` | Load pretrained tokenizer | Tokenizer |
| `to_str()` | IO | `pretty=False` | Convert tokenizer to JSON string | str |
| `post_process()` | Processing | `encoding, pair=None, add_special_tokens=True` | Apply post-processing | Encoding |
| `num_special_tokens_to_add()` | Info | `is_pair` | Get number of special tokens | int |

## Model Classes and Methods

### BPE (Byte Pair Encoding)

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | `vocab=None, merges=None, cache_capacity=10000, dropout=None, unk_token=None, continuing_subword_prefix=None, end_of_word_suffix=None, fuse_unk=False, byte_fallback=False` | Initialize BPE model | BPE |
| `from_file()` | IO | `vocab, merges, **kwargs` | Load BPE from files | BPE |
| `tokenize()` | Core | `sequence` | Tokenize sequence | List[Token] |
| `token_to_id()` | Vocab | `token` | Get token ID | int or None |
| `id_to_token()` | Vocab | `id` | Get token string | str or None |
| `get_vocab()` | Vocab | None | Get vocabulary | Dict[str, int] |
| `get_vocab_size()` | Vocab | None | Get vocabulary size | int |
| `save()` | IO | `folder, prefix=None` | Save model files | List[str] |

### WordPiece

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | `vocab=None, unk_token='[UNK]', max_input_chars_per_word=100, continuing_subword_prefix='##'` | Initialize WordPiece model | WordPiece |
| `from_file()` | IO | `vocab, **kwargs` | Load WordPiece from file | WordPiece |
| `tokenize()` | Core | `sequence` | Tokenize sequence | List[Token] |
| `token_to_id()` | Vocab | `token` | Get token ID | int or None |
| `id_to_token()` | Vocab | `id` | Get token string | str or None |
| `get_vocab()` | Vocab | None | Get vocabulary | Dict[str, int] |
| `get_vocab_size()` | Vocab | None | Get vocabulary size | int |
| `save()` | IO | `folder, prefix=None` | Save model files | List[str] |

### Unigram

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | `vocab=None, unk_id=0, byte_fallback=False` | Initialize Unigram model | Unigram |
| `tokenize()` | Core | `sequence` | Tokenize sequence | List[Token] |
| `token_to_id()` | Vocab | `token` | Get token ID | int or None |
| `id_to_token()` | Vocab | `id` | Get token string | str or None |
| `get_vocab()` | Vocab | None | Get vocabulary | Dict[str, int] |
| `get_vocab_size()` | Vocab | None | Get vocabulary size | int |
| `save()` | IO | `folder, prefix=None` | Save model files | List[str] |

### WordLevel

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | `vocab=None, unk_token='[UNK]'` | Initialize WordLevel model | WordLevel |
| `from_file()` | IO | `vocab, **kwargs` | Load from vocabulary file | WordLevel |
| `tokenize()` | Core | `sequence` | Tokenize sequence | List[Token] |
| `token_to_id()` | Vocab | `token` | Get token ID | int or None |
| `id_to_token()` | Vocab | `id` | Get token string | str or None |
| `get_vocab()` | Vocab | None | Get vocabulary | Dict[str, int] |
| `get_vocab_size()` | Vocab | None | Get vocabulary size | int |
| `save()` | IO | `folder, prefix=None` | Save model files | List[str] |

## Pre-tokenizer Classes and Methods

### Whitespace

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | None | Initialize whitespace pre-tokenizer | Whitespace |
| `pre_tokenize()` | Core | `pretok` | Pre-tokenize on whitespace | List[Tuple[str, Tuple[int, int]]] |

### WhitespaceSplit

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | None | Initialize whitespace split pre-tokenizer | WhitespaceSplit |
| `pre_tokenize()` | Core | `pretok` | Pre-tokenize splitting on whitespace | List[Tuple[str, Tuple[int, int]]] |

### Punctuation

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | `behavior='isolated'` | Initialize punctuation pre-tokenizer | Punctuation |
| `pre_tokenize()` | Core | `pretok` | Pre-tokenize handling punctuation | List[Tuple[str, Tuple[int, int]]] |

### Sequence

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | `pretokenizers` | Initialize sequence of pre-tokenizers | Sequence |
| `pre_tokenize()` | Core | `pretok` | Apply pre-tokenizers in sequence | List[Tuple[str, Tuple[int, int]]] |

### ByteLevel

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | `add_prefix_space=True, trim_offsets=True, use_regex=True` | Initialize byte-level pre-tokenizer | ByteLevel |
| `pre_tokenize()` | Core | `pretok` | Pre-tokenize at byte level | List[Tuple[str, Tuple[int, int]]] |
| `alphabet()` | Static | None | Get byte-level alphabet | List[str] |

### Split

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | `pattern, behavior='removed', invert=False` | Initialize split pre-tokenizer | Split |
| `pre_tokenize()` | Core | `pretok` | Pre-tokenize with pattern splitting | List[Tuple[str, Tuple[int, int]]] |

### Digits

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | `individual_digits=False` | Initialize digits pre-tokenizer | Digits |
| `pre_tokenize()` | Core | `pretok` | Pre-tokenize handling digits | List[Tuple[str, Tuple[int, int]]] |

### Metaspace

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | `replacement='▁', add_prefix_space=True` | Initialize metaspace pre-tokenizer | Metaspace |
| `pre_tokenize()` | Core | `pretok` | Pre-tokenize with metaspace | List[Tuple[str, Tuple[int, int]]] |

### CharDelimiterSplit

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | `delimiter` | Initialize character delimiter split | CharDelimiterSplit |
| `pre_tokenize()` | Core | `pretok` | Pre-tokenize splitting on delimiter | List[Tuple[str, Tuple[int, int]]] |

## Normalizer Classes and Methods

### NFD

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | None | Initialize NFD normalizer | NFD |
| `normalize_str()` | Core | `sequence` | Apply NFD normalization | str |

### NFKD

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | None | Initialize NFKD normalizer | NFKD |
| `normalize_str()` | Core | `sequence` | Apply NFKD normalization | str |

### NFC

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | None | Initialize NFC normalizer | NFC |
| `normalize_str()` | Core | `sequence` | Apply NFC normalization | str |

### NFKC

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | None | Initialize NFKC normalizer | NFKC |
| `normalize_str()` | Core | `sequence` | Apply NFKC normalization | str |

### Lowercase

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | None | Initialize lowercase normalizer | Lowercase |
| `normalize_str()` | Core | `sequence` | Convert to lowercase | str |

### Strip

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | `left=True, right=True` | Initialize strip normalizer | Strip |
| `normalize_str()` | Core | `sequence` | Strip whitespace | str |

### StripAccents

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | None | Initialize accent stripping normalizer | StripAccents |
| `normalize_str()` | Core | `sequence` | Remove accents | str |

### Replace

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | `pattern, content` | Initialize replace normalizer | Replace |
| `normalize_str()` | Core | `sequence` | Replace pattern with content | str |

### BertNormalizer

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | `clean_text=True, handle_chinese_chars=True, strip_accents=None, lowercase=True` | Initialize BERT normalizer | BertNormalizer |
| `normalize_str()` | Core | `sequence` | Apply BERT normalization | str |

### Precompiled

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | `precompiled_charsmap` | Initialize precompiled normalizer | Precompiled |
| `normalize_str()` | Core | `sequence` | Apply precompiled normalization | str |

### Sequence (Normalizer)

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | `normalizers` | Initialize sequence of normalizers | Sequence |
| `normalize_str()` | Core | `sequence` | Apply normalizers in sequence | str |

## Post-Processor Classes and Methods

### BertProcessing

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | `sep=('[SEP]', 102), cls=('[CLS]', 101)` | Initialize BERT post-processor | BertProcessing |
| `process()` | Core | `encoding, pair=None, add_special_tokens=True` | Apply BERT post-processing | Encoding |
| `num_special_tokens_to_add()` | Info | `is_pair` | Get number of special tokens | int |

### RobertaProcessing

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | `sep=('</s>', 2), cls=('<s>', 0), trim_offsets=True, add_prefix_space=True` | Initialize RoBERTa post-processor | RobertaProcessing |
| `process()` | Core | `encoding, pair=None, add_special_tokens=True` | Apply RoBERTa post-processing | Encoding |
| `num_special_tokens_to_add()` | Info | `is_pair` | Get number of special tokens | int |

### ByteLevel (Post-processor)

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | `trim_offsets=True` | Initialize byte-level post-processor | ByteLevel |
| `process()` | Core | `encoding, pair=None, add_special_tokens=True` | Apply byte-level post-processing | Encoding |

### TemplateProcessing

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | `single, pair=None, special_tokens=None` | Initialize template post-processor | TemplateProcessing |
| `process()` | Core | `encoding, pair=None, add_special_tokens=True` | Apply template post-processing | Encoding |
| `num_special_tokens_to_add()` | Info | `is_pair` | Get number of special tokens | int |

### Sequence (Post-processor)

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | `processors` | Initialize sequence of post-processors | Sequence |
| `process()` | Core | `encoding, pair=None, add_special_tokens=True` | Apply post-processors in sequence | Encoding |

## Decoder Classes and Methods

### ByteLevel (Decoder)

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | None | Initialize byte-level decoder | ByteLevel |
| `decode()` | Core | `tokens` | Decode tokens to string | str |

### WordPiece (Decoder)

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | `prefix='##', cleanup=True` | Initialize WordPiece decoder | WordPiece |
| `decode()` | Core | `tokens` | Decode WordPiece tokens | str |

### Metaspace (Decoder)

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | `replacement='▁', add_prefix_space=True` | Initialize Metaspace decoder | Metaspace |
| `decode()` | Core | `tokens` | Decode Metaspace tokens | str |

### BPEDecoder

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | `suffix='</w>'` | Initialize BPE decoder | BPEDecoder |
| `decode()` | Core | `tokens` | Decode BPE tokens | str |

### CTC

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | `pad_token='<pad>', word_delimiter_token='|', cleanup=True` | Initialize CTC decoder | CTC |
| `decode()` | Core | `tokens` | Decode CTC tokens | str |

### Sequence (Decoder)

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | `decoders` | Initialize sequence of decoders | Sequence |
| `decode()` | Core | `tokens` | Apply decoders in sequence | str |

## Trainer Classes and Methods

### BpeTrainer

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | `vocab_size=30000, min_frequency=0, special_tokens=None, limit_alphabet=None, initial_alphabet=None, continuing_subword_prefix=None, end_of_word_suffix=None, show_progress=True` | Initialize BPE trainer | BpeTrainer |

### WordPieceTrainer

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | `vocab_size=30000, min_frequency=0, special_tokens=None, limit_alphabet=None, initial_alphabet=None, continuing_subword_prefix='##', show_progress=True` | Initialize WordPiece trainer | WordPieceTrainer |

### WordLevelTrainer

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | `vocab_size=30000, min_frequency=0, special_tokens=None, show_progress=True` | Initialize WordLevel trainer | WordLevelTrainer |

### UnigramTrainer

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | `vocab_size=8000, special_tokens=None, shrinking_factor=0.75, unk_token=None, max_piece_length=16, n_sub_iterations=2, show_progress=True` | Initialize Unigram trainer | UnigramTrainer |

## Encoding Class Methods

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | `ids, type_ids=None, tokens=None, words=None, offsets=None, special_tokens_mask=None, attention_mask=None, overflowing=None, sequence_ranges=None` | Initialize encoding | Encoding |
| `char_to_token()` | Mapping | `char_pos, sequence_index=0` | Map character position to token | int or None |
| `char_to_word()` | Mapping | `char_pos, sequence_index=0` | Map character position to word | int or None |
| `token_to_chars()` | Mapping | `token_index, sequence_index=0` | Map token to character span | Tuple[int, int] or None |
| `token_to_sequence()` | Mapping | `token_index` | Map token to sequence index | int or None |
| `token_to_word()` | Mapping | `token_index, sequence_index=0` | Map token to word index | int or None |
| `word_to_chars()` | Mapping | `word_index, sequence_index=0` | Map word to character span | Tuple[int, int] or None |
| `word_to_tokens()` | Mapping | `word_index, sequence_index=0` | Map word to token span | Tuple[int, int] or None |
| `merge()` | Transform | `encoding, growing_offsets=True` | Merge with another encoding | Encoding |
| `pad()` | Transform | `length, direction='right', pad_id=0, pad_type_id=0, pad_token='[PAD]'` | Pad encoding to length | None |
| `truncate()` | Transform | `max_length, stride=0, direction='right'` | Truncate encoding | List[Encoding] |
| `n_sequences` | Property | None | Number of sequences | int |
| `ids` | Property | None | Token IDs | List[int] |
| `tokens` | Property | None | Token strings | List[str] |
| `words` | Property | None | Word indices | List[int or None] |
| `type_ids` | Property | None | Type IDs | List[int] |
| `offsets` | Property | None | Character offsets | List[Tuple[int, int]] |
| `special_tokens_mask` | Property | None | Special tokens mask | List[int] |
| `attention_mask` | Property | None | Attention mask | List[int] |
| `overflowing` | Property | None | Overflowing tokens | List[Encoding] |

## Utility Functions and Classes

### AddedToken

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | `content, single_word=False, lstrip=False, rstrip=False, normalized=True, special=False` | Initialize added token | AddedToken |
| `__str__()` | Property | None | String representation | str |
| `content` | Property | None | Token content | str |
| `lstrip` | Property | None | Left strip flag | bool |
| `rstrip` | Property | None | Right strip flag | bool |
| `single_word` | Property | None | Single word flag | bool |
| `normalized` | Property | None | Normalized flag | bool |
| `special` | Property | None | Special token flag | bool |

### Token

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | `id, value, offsets` | Initialize token | Token |
| `id` | Property | None | Token ID | int |
| `value` | Property | None | Token value | str |
| `offsets` | Property | None | Character offsets | Tuple[int, int] |

### PreTokenizedString

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__init__()` | Constructor | `sequence` | Initialize pre-tokenized string | PreTokenizedString |
| `split()` | Transform | `delimiter` | Split on delimiter | None |
| `normalize()` | Transform | `normalizer` | Apply normalizer | None |
| `tokenize()` | Transform | `tokenizer` | Apply tokenizer | None |
| `to_encoding()` | Transform | `type_id=0, word_idx=None` | Convert to encoding | Encoding |
| `get_splits()` | Access | None | Get splits | List[NormalizedString] |

---

## Detailed Method Explanations

### Core Tokenization Process

**`__call__()`** is the main entry point that handles the complete tokenization pipeline. It accepts text input and applies all configured components (normalization, pre-tokenization, model tokenization, and post-processing) to produce the final encoding.

**`encode()`** and **`encode_batch()`** provide more direct access to the tokenization process, returning Encoding objects that contain detailed information about the tokenization including token IDs, offsets, and mappings.

### Training Process

The training process involves creating a trainer object (BpeTrainer, WordPieceTrainer, etc.) and calling the `train()` method on the tokenizer. The tokenizer can be trained using files or iterators, with options for vocabulary size, minimum frequency, and special tokens.

### Component Architecture

The tokenizers library uses a modular architecture where each tokenizer consists of four main components:

1. **Normalizer**: Cleans and standardizes input text
2. **Pre-tokenizer**: Splits text into smaller units
3. **Model**: The actual tokenization algorithm (BPE, WordPiece, etc.)
4. **Post-processor**: Adds special tokens and formats the output

### Advanced Features

**Mapping Methods**: The Encoding class provides detailed mapping capabilities between characters, words, and tokens, enabling precise alignment for tasks like named entity recognition.

**Padding and Truncation**: Built-in support for batching sequences of different lengths with configurable padding and truncation strategies.

**Special Token Handling**: Comprehensive support for special tokens with configurable behavior for normalization and tokenization.

This comprehensive reference covers all methods available in the Hugging Face tokenizers library, providing a complete guide for tokenization tasks in NLP applications.