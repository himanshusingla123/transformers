# Complete Hugging Face Datasets Library Methods Reference

This comprehensive table covers all methods available in the Hugging Face `datasets` library, organized by category.

## Core Loading Functions

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `load_dataset()` | Core | `path, name=None, data_dir=None, data_files=None, split=None, cache_dir=None, features=None, download_config=None, download_mode=None, verification_mode=None, ignore_verifications=False, keep_in_memory=False, save_infos=False, revision=None, token=None, use_auth_token=None, task=None, streaming=False, num_proc=None, storage_options=None, trust_remote_code=None, **config_kwargs` | Main function to load datasets from Hub or local files | Dataset or DatasetDict |
| `load_from_disk()` | Core | `dataset_path, fs=None, keep_in_memory=None, storage_options=None` | Load a dataset from a directory on disk | Dataset or DatasetDict |
| `load_dataset_builder()` | Core | `path, name=None, data_dir=None, data_files=None, cache_dir=None, features=None, download_config=None, download_mode=None, revision=None, token=None, use_auth_token=None, **config_kwargs` | Load dataset builder without downloading data | DatasetBuilder |
| `get_dataset_config_names()` | Core | `path, revision=None, download_config=None, download_mode=None, dynamic_modules_path=None, data_files=None, **download_kwargs` | Get list of available configurations for a dataset | List[str] |
| `get_dataset_split_names()` | Core | `path, config_name=None, data_dir=None, data_files=None, download_config=None, download_mode=None, revision=None, token=None, use_auth_token=None, **config_kwargs` | Get list of available splits for a dataset | List[str] |

## Dataset Class Methods

### Basic Operations

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `__len__()` | Basic | `self` | Get number of rows in dataset | int |
| `__getitem__()` | Basic | `self, key` | Get item(s) from dataset by index/slice | Dict or List[Dict] |
| `__iter__()` | Basic | `self` | Iterate over dataset rows | Iterator |
| `__repr__()` | Basic | `self` | String representation of dataset | str |
| `shape` | Basic | Property | Get shape of dataset (rows, columns) | Tuple[int, int] |
| `column_names` | Basic | Property | Get list of column names | List[str] |
| `features` | Basic | Property | Get dataset features schema | Features |
| `num_columns` | Basic | Property | Get number of columns | int |
| `num_rows` | Basic | Property | Get number of rows | int |
| `cache_files` | Basic | Property | Get list of cache files | List[Dict] |
| `dataset_size` | Basic | Property | Get dataset size in bytes | int |

### Data Manipulation

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `map()` | Transform | `function, with_indices=False, with_rank=False, input_columns=None, batched=False, batch_size=1000, drop_last_batch=False, remove_columns=None, keep_in_memory=False, load_from_cache_file=None, cache_file_name=None, writer_batch_size=1000, features=None, disable_nullable=False, fn_kwargs=None, num_proc=None, suffix_template='_{rank:05d}_of_{num_proc:05d}', new_fingerprint=None, desc=None` | Apply function to every example | Dataset |
| `select()` | Transform | `indices, keep_in_memory=False, indices_cache_file_name=None, writer_batch_size=1000, new_fingerprint=None` | Select examples by indices | Dataset |
| `select_columns()` | Transform | `column_names` | Select specific columns | Dataset |
| `remove_columns()` | Transform | `column_names` | Remove specific columns | Dataset |
| `rename_column()` | Transform | `original_column_name, new_column_name` | Rename a column | Dataset |
| `rename_columns()` | Transform | `column_mapping` | Rename multiple columns | Dataset |
| `cast_column()` | Transform | `column, feature, new_fingerprint=None` | Cast column to new feature type | Dataset |
| `cast()` | Transform | `features, batch_size=1000, keep_in_memory=False, load_from_cache_file=None, cache_file_name=None, writer_batch_size=1000, num_proc=None, new_fingerprint=None, desc=None` | Cast dataset to new features schema | Dataset |
| `add_column()` | Transform | `name, column, new_fingerprint=None` | Add a new column | Dataset |
| `concatenate_datasets()` | Transform | `datasets, info=None, split=None, axis=0` | Concatenate multiple datasets | Dataset |

### Filtering and Sorting

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `filter()` | Filter | `function, with_indices=False, with_rank=False, input_columns=None, batched=False, batch_size=1000, keep_in_memory=False, load_from_cache_file=None, cache_file_name=None, writer_batch_size=1000, fn_kwargs=None, num_proc=None, new_fingerprint=None, desc=None` | Filter examples based on condition | Dataset |
| `sort()` | Sort | `column, reverse=False, kind=None, null_placement='at_end', keep_in_memory=False, load_from_cache_file=None, indices_cache_file_name=None, writer_batch_size=1000, new_fingerprint=None` | Sort dataset by column values | Dataset |
| `shuffle()` | Sort | `seed=None, generator=None, keep_in_memory=False, load_from_cache_file=None, indices_cache_file_name=None, writer_batch_size=1000, new_fingerprint=None` | Shuffle dataset rows | Dataset |
| `train_test_split()` | Split | `test_size=None, train_size=None, shuffle=True, seed=None, generator=None, stratify_by_column=None, keep_in_memory=False, load_from_cache_file=None, train_indices_cache_file_name=None, test_indices_cache_file_name=None, writer_batch_size=1000, train_new_fingerprint=None, test_new_fingerprint=None` | Split dataset into train/test | DatasetDict |
| `shard()` | Split | `num_shards, index, contiguous=False, keep_in_memory=False, indices_cache_file_name=None, writer_batch_size=1000` | Get a shard of the dataset | Dataset |

### Data Access and Iteration

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `to_pandas()` | Export | `batch_size=None, batched=False` | Convert to pandas DataFrame | pd.DataFrame |
| `to_dict()` | Export | `batch_size=None, batched=False` | Convert to dictionary | Dict |
| `to_list()` | Export | `batch_size=None, batched=False` | Convert to list of examples | List[Dict] |
| `iter()` | Access | `batch_size, drop_last_batch=False` | Iterate in batches | Iterator |
| `take()` | Access | `n` | Take first n examples | Dataset |
| `skip()` | Access | `n` | Skip first n examples | Dataset |
| `with_format()` | Format | `type=None, columns=None, output_all_columns=False, **format_kwargs` | Set output format (numpy, torch, tf, pandas) | Dataset |
| `set_format()` | Format | `type=None, columns=None, output_all_columns=False, **format_kwargs` | Set output format in-place | None |
| `reset_format()` | Format | `self` | Reset to original format | None |
| `formatted_as()` | Format | `type=None, columns=None, output_all_columns=False, **format_kwargs` | Context manager for temporary format | ContextManager |

### Search and Information

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `search()` | Search | `query, k=None` | Search examples using query | Dataset |
| `search_batch()` | Search | `queries, k=None` | Search multiple queries | List[Dataset] |
| `get_nearest_examples()` | Search | `query, k=None` | Get nearest examples to query | Dict |
| `get_nearest_examples_batch()` | Search | `queries, k=None` | Get nearest examples for multiple queries | Dict |
| `add_faiss_index()` | Search | `column, index_name=None, device=None, string_factory=None, metric_type=None, custom_index=None, train_size=None, faiss_verbose=False` | Add FAISS index for similarity search | None |
| `add_faiss_index_from_external_arrays()` | Search | `external_arrays, index_name, index=None, device=None, string_factory=None, metric_type=None, custom_index=None, faiss_verbose=False` | Add FAISS index from external arrays | None |
| `save_faiss_index()` | Search | `index_name, file` | Save FAISS index to file | None |
| `load_faiss_index()` | Search | `index_name, file, device=None` | Load FAISS index from file | None |
| `drop_index()` | Search | `index_name` | Drop search index | None |
| `list_indexes()` | Search | `self` | List available indexes | List[str] |
| `info` | Info | Property | Get dataset info | DatasetInfo |
| `split` | Info | Property | Get split name | str |
| `builder_name` | Info | Property | Get builder name | str |
| `config_name` | Info | Property | Get config name | str |
| `version` | Info | Property | Get dataset version | Version |

### Saving and Loading

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `save_to_disk()` | Save | `dataset_path, fs=None, max_shard_size=None, num_shards=None, num_proc=None, storage_options=None` | Save dataset to disk | None |
| `to_csv()` | Export | `path_or_buf=None, batch_size=None, num_proc=None, **to_csv_kwargs` | Export to CSV format | str or None |
| `to_json()` | Export | `path_or_buf=None, batch_size=None, num_proc=None, **to_json_kwargs` | Export to JSON format | str or None |
| `to_parquet()` | Export | `path_or_buf=None, batch_size=None, **parquet_writer_kwargs` | Export to Parquet format | str or None |
| `to_sql()` | Export | `name, con, batch_size=None, **sql_writer_kwargs` | Export to SQL database | None |
| `push_to_hub()` | Hub | `repo_id, config_name=None, split=None, data_dir=None, commit_message=None, commit_description=None, private=None, token=None, revision=None, create_pr=False, max_shard_size=None, num_shards=None, embed_external_files=True` | Push dataset to Hub | str |

### Advanced Operations

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `flatten()` | Transform | `new_fingerprint=None, max_depth=16` | Flatten nested structures | Dataset |
| `class_encode_column()` | Transform | `column` | Encode column as ClassLabel | Dataset |
| `prepare_for_task()` | Transform | `task, id=0` | Prepare dataset for specific task | Dataset |
| `cleanup_cache_files()` | Cache | `self` | Clean up cache files | int |
| `unique()` | Stats | `column` | Get unique values in column | List |
| `set_transform()` | Transform | `transform, columns=None, output_all_columns=False` | Set transform function | None |
| `with_transform()` | Transform | `transform, columns=None, output_all_columns=False` | Apply transform function | Dataset |
| `add_item()` | Modify | `item, new_fingerprint=None` | Add single item to dataset | Dataset |

## DatasetDict Methods

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `keys()` | Access | `self` | Get split names | KeysView |
| `values()` | Access | `self` | Get datasets | ValuesView |
| `items()` | Access | `self` | Get (split, dataset) pairs | ItemsView |
| `map()` | Transform | `function, with_indices=False, with_rank=False, input_columns=None, batched=False, batch_size=1000, drop_last_batch=False, remove_columns=None, keep_in_memory=False, load_from_cache_file=None, cache_file_names=None, writer_batch_size=1000, features=None, disable_nullable=False, fn_kwargs=None, num_proc=None, new_fingerprint=None, desc=None` | Apply function to all splits | DatasetDict |
| `filter()` | Filter | `function, with_indices=False, with_rank=False, input_columns=None, batched=False, batch_size=1000, keep_in_memory=False, load_from_cache_file=None, cache_file_names=None, writer_batch_size=1000, fn_kwargs=None, num_proc=None, desc=None` | Filter all splits | DatasetDict |
| `sort()` | Sort | `column, reverse=False, kind=None, null_placement='at_end', keep_in_memory=False, load_from_cache_file=None, indices_cache_file_names=None, writer_batch_size=1000` | Sort all splits | DatasetDict |
| `shuffle()` | Sort | `seeds=None, seed=None, generators=None, generator=None, keep_in_memory=False, load_from_cache_file=None, indices_cache_file_names=None, writer_batch_size=1000` | Shuffle all splits | DatasetDict |
| `set_format()` | Format | `type=None, columns=None, output_all_columns=False, **format_kwargs` | Set format for all splits | None |
| `reset_format()` | Format | `self` | Reset format for all splits | None |
| `with_format()` | Format | `type=None, columns=None, output_all_columns=False, **format_kwargs` | Context manager for format | ContextManager |
| `cast()` | Transform | `features, batch_size=1000, keep_in_memory=False, load_from_cache_file=None, cache_file_names=None, writer_batch_size=1000, num_proc=None` | Cast all splits | DatasetDict |
| `cast_column()` | Transform | `column, feature` | Cast column in all splits | DatasetDict |
| `remove_columns()` | Transform | `column_names` | Remove columns from all splits | DatasetDict |
| `rename_column()` | Transform | `original_column_name, new_column_name` | Rename column in all splits | DatasetDict |
| `rename_columns()` | Transform | `column_mapping` | Rename columns in all splits | DatasetDict |
| `select_columns()` | Transform | `column_names` | Select columns in all splits | DatasetDict |
| `save_to_disk()` | Save | `dataset_dict_path, fs=None, max_shard_size=None, num_shards=None, num_proc=None, storage_options=None` | Save all splits to disk | None |
| `load_from_disk()` | Load | `dataset_dict_path, fs=None, keep_in_memory=None, storage_options=None` | Load all splits from disk | DatasetDict |
| `push_to_hub()` | Hub | `repo_id, config_name=None, commit_message=None, commit_description=None, private=None, token=None, revision=None, create_pr=False, max_shard_size=None, num_shards=None, embed_external_files=True` | Push all splits to Hub | str |
| `cleanup_cache_files()` | Cache | `self` | Clean cache for all splits | Dict[str, int] |
| `unique()` | Stats | `column` | Get unique values for all splits | Dict[str, List] |
| `flatten()` | Transform | `max_depth=16` | Flatten all splits | DatasetDict |
| `prepare_for_task()` | Transform | `task, id=0` | Prepare all splits for task | DatasetDict |

## Streaming Dataset Methods

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `take()` | Access | `n` | Take first n examples | IterableDataset |
| `skip()` | Access | `n` | Skip first n examples | IterableDataset |
| `shuffle()` | Transform | `seed=None, generator=None, buffer_size=1000` | Shuffle streaming dataset | IterableDataset |
| `map()` | Transform | `function, with_indices=False, with_rank=False, input_columns=None, batched=False, batch_size=1000, drop_last_batch=False, remove_columns=None, fn_kwargs=None` | Apply function to streaming data | IterableDataset |
| `filter()` | Filter | `function, with_indices=False, with_rank=False, input_columns=None, batched=False, batch_size=1000, fn_kwargs=None` | Filter streaming data | IterableDataset |
| `select_columns()` | Transform | `column_names` | Select columns in stream | IterableDataset |
| `remove_columns()` | Transform | `column_names` | Remove columns from stream | IterableDataset |
| `rename_column()` | Transform | `original_column_name, new_column_name` | Rename column in stream | IterableDataset |
| `rename_columns()` | Transform | `column_mapping` | Rename columns in stream | IterableDataset |
| `cast_column()` | Transform | `column, feature` | Cast column type in stream | IterableDataset |
| `cast()` | Transform | `features` | Cast features in stream | IterableDataset |
| `with_format()` | Format | `type=None, columns=None, output_all_columns=False, **format_kwargs` | Set format for stream | IterableDataset |
| `set_epoch()` | Control | `epoch` | Set epoch for reproducible shuffling | None |

## Utility Functions

| Function | Parameters | Description | Return Type |
|----------|------------|-------------|-------------|
| `enable_caching()` | `cache_dir=None` | Enable dataset caching | None |
| `disable_caching()` | None | Disable dataset caching | None |
| `is_caching_enabled()` | None | Check if caching is enabled | bool |
| `set_caching_enabled()` | `cache_enabled` | Set caching state | None |
| `fingerprint.Hasher()` | Various | Create fingerprint hasher | Hasher |
| `disable_progress_bar()` | None | Disable progress bars | None |
| `enable_progress_bar()` | None | Enable progress bars | None |
| `get_dataset_infos()` | `path, data_files=None, download_config=None, download_mode=None, revision=None, token=None, use_auth_token=None, **config_kwargs` | Get dataset information | Dict |
| `inspect_dataset()` | `path, name=None, data_files=None, cache_dir=None, **config_kwargs` | Inspect dataset structure | DatasetInfo |
| `list_datasets()` | `with_community_datasets=True, with_details=False` | List available datasets | List[str] |

## Audio Processing Methods

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `cast_column()` | Audio | `column, Audio(sampling_rate=None, mono=True, decode=True, id=None)` | Cast column to Audio type | Dataset |

## Image Processing Methods  

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `cast_column()` | Image | `column, Image(mode=None, decode=True, id=None)` | Cast column to Image type | Dataset |

## Text Processing Methods

| Method | Category | Parameters | Description | Return Type |
|--------|----------|------------|-------------|-------------|
| `cast_column()` | Text | `column, Value(dtype='string', id=None)` | Cast column to text type | Dataset |
| `cast_column()` | Labels | `column, ClassLabel(names=None, names_file=None, num_classes=None, id=None)` | Cast column to ClassLabel type | Dataset |

---

## Detailed Method Explanations

### Core Loading Functions

**`load_dataset()`** is the primary entry point for loading datasets. It can load from the Hugging Face Hub, local files, or custom dataset scripts. The `streaming=True` parameter allows for memory-efficient processing of large datasets.

**`load_from_disk()`** restores previously saved datasets from local storage, preserving all metadata and indices.

**`load_dataset_builder()`** provides access to dataset metadata without downloading the actual data, useful for inspection.

### Data Manipulation

**`map()`** is the most powerful transformation method, allowing custom functions to be applied to every example. It supports parallel processing via `num_proc` and can operate in batched mode for efficiency.

**`filter()`** removes examples that don't meet specified criteria. Like `map()`, it supports parallel processing and batched operations.

**`select()`** creates a subset using specific indices, while `select_columns()` and `remove_columns()` manage column structure.

### Advanced Features

**FAISS Integration**: Methods like `add_faiss_index()` enable efficient similarity search on large datasets using Facebook's FAISS library.

**Streaming Support**: The `IterableDataset` class provides memory-efficient processing for datasets too large to fit in RAM.

**Hub Integration**: `push_to_hub()` allows easy sharing of processed datasets with the community.

**Format Support**: The library supports multiple output formats (PyTorch, TensorFlow, NumPy, Pandas) via `set_format()` and `with_format()`.

This table covers the complete API surface of the Hugging Face datasets library as of early 2024. Each method is designed to work efficiently with large-scale datasets while maintaining a simple, intuitive interface.