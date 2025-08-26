**Decoder-only models**, **auto-regressive** aur **causal attention** meaning:

---

### 🔹 1. Transformer Encoder vs Decoder

* **Encoder models** (jaise BERT) → input ke **poore sentence ke words ko ek saath dekhte hain** (bidirectional). Matlab word ke left aur right dono context available hote hain.
* **Decoder models** (jaise GPT, LLaMA, Falcon) → sirf **pichle words** dekh sakte hain, future words nahi.

---

### 🔹 2. Auto-regressive Model

"Auto-regressive" ka matlab hai:

* Model **ek word ek word karke predict karta hai**.
* Har step pe model ko **sirf ab tak generate kiye gaye words ka context** milta hai, aur usi base pe agla word generate hota hai.
* Example:

  ```
  Input so far: "The cat sat on the"
  Model predicts: "mat"
  ```

  Agle step pe input ban jata hai `"The cat sat on the mat"`, fir next word predict karega.

Isko **causal (left-to-right) generation** bhi bolte hain.

---

### 🔹 3. Attention Restriction

Decoder-only model ke attention layers mein **masking hota hai**:

* Har word **sirf apne se pehle wale words ko dekh sakta hai**.
* Matlab agar sentence hai `"The cat sat on the mat"`, to word `"sat"` ke attention ko `"on"`, `"the"`, `"mat"` ke embeddings nahi milenge, sirf `"The"` aur `"cat"` ke milenge.
* Yeh ensure karta hai ki model **future words ko cheat karke use nahi kare** prediction ke liye.

---

### 🔹 4. Why called Auto-Regressive?

"Auto" = apna output khud hi input ban jata hai next step ke liye.
"Regressive" = ek step pichhle step ke upar depend karta hai.

Jaise ek chain reaction:

```
y1 = f(x)
y2 = f(x, y1)
y3 = f(x, y1, y2)
...
```

LLMs (GPT, LLaMA, Mistral) isi principle pe kaam karte hain.

---

### 🔹 5. Simple Analogy (Romanized Hindi)

Socho tum ek story likh rahe ho:

* Tumhe sirf **ab tak likha hua text** yaad hai.
* Tum us base pe decide karte ho ki agla word kya ho.
* Tum future ke words (jo tum abhi likhne wale ho) nahi dekh sakte.
  👉 Yeh hi hai **auto-regressive generation**.

---

### ✅ Quick Example (PyTorch + GPT2)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

input_text = "The cat sat on the"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=10)

print(tokenizer.decode(outputs[0]))
```

Yeh GPT2 (decoder-only, auto-regressive) **"The cat sat on the mat"** jaisa kuch generate karega.

---

👉 Kya tum chahte ho main **encoder-only vs decoder-only vs encoder-decoder** models ka ek comparison chart bana dun (BERT vs GPT vs T5 style)?
