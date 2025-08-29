# from datasets import Dataset

# # Example dataset
# data = {"text": ["I love NLP", "Transformers are powerful"]}
# dataset = Dataset.from_dict(data)

# print("Original dataset:")
# print(dataset)

# # ---------------------------
# # 1. Normal one-to-one mapping
# # ---------------------------
# def add_length(example):
#     return {"text": example["text"], "length": len(example["text"].split())}

# one_to_one = dataset.map(add_length)
# print("\nOne-to-one mapping result:")
# print(one_to_one)
# print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
# # ---------------------------
# # 2. One-to-many mapping (chunking text)
# # ---------------------------
# def split_words(example):
#     words = example["text"].split()
#     print("****************",words[1], "*********")
#     print("-------------")
#     # split into chunks of 2 words
#     chunks = [" ".join(words[i:i+2]) for i in range(0, len(words), 2)]
#     print(chunks)
#     return {"text_chunk": chunks}

# one_to_many = dataset.map(split_words, batched=False, remove_columns=["text"])
# print("\nOne-to-many mapping result:")
# print(one_to_many)
# print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

# # ---------------------------
# # 3. One-to-zero mapping (filtering)
# # ---------------------------
# def filter_short(example):
#     if len(example["text"].split()) > 2:
#         return {"text": example["text"]}
#     else:
#         return {}  # no output

# one_to_zero = dataset.map(filter_short, remove_columns=["text"])
# print("\nOne-to-zero mapping result:")
# print(one_to_zero['text'])



a = [(0 , 1) , (2,3) , (4,5)]
for i , offset in enumerate(a):
    print(i)
    print(offset)

examples = {
    "question":
        ["helo" , "world"]
}
questions = [q.strip() for q in examples["question"]]
print(questions)

offset_mapping =[(0,0), (0,3), (4,9), (10,16), (16,17), (0,0),
     (0,6), (7,10), (11,18), (19,21), (22,29), (30,41),
     (42,44), (45,48), (49,54), (55,61), (61,62),
     (0,0), (0,0)]

offset = offset_mapping[5]
for k, o in enumerate(offset):
    print(k , o)