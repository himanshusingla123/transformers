## 🔹 Problem in Normal Attention

* Standard Attention me hum ye karte hain:

  $$
  \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
  $$
* Yaha **QKᵀ** ek **n × n** matrix hota hai (agar sequence length = n ho).
* Complexity: **O(n²)** → bahut mehenga ho jata hai jab sequence bada ho (1M tokens etc).

---

## 🔹 Key Idea of Reformer (LSH Attention)

* Observation: Softmax(QKᵀ) me **sirf bade similarity values matter karte hain**.
  Matlab query $q$ sirf unhi keys $k$ me interested hai jo uske **close / similar** hain.
* To **sabhi pairs (q,k)** check karna zaroori nahi hai.
* Reformer isko optimize karta hai using **Locality-Sensitive Hashing (LSH)**.

---

## 🔹 How LSH Attention Works

1. **Hashing Queries & Keys**

   * Ek hash function lagate hain jo vector space me **similar vectors ko same bucket me daale**.
   * Matlab agar $q$ aur $k$ close hain, toh high chance hai dono ek bucket me gaye.
   * Buckets ban jaate hain → ab hum sirf **bucket ke andar wali Q–K pairs** check karte hain.

2. **Masking Trick**

   * Agar $q = k$ (matlab same token), toh woh similarity artificially bohot high ho jaati hai.
   * Isliye self-token ko mask kar dete hain (except first position), warna model khud ko hi attend karta rahega.

3. **Multiple Rounds (n\_rounds)**

   * Hashing thoda random hota hai. Kabhi kabhi close vectors alag bucket me chale jaate hain.
   * To multiple hash functions use karte hain (say 2–4 rounds).
   * Har round me alag bucketization → phir unke results ko **average** kar dete hain.

---

## 🔹 Complexity Improvement

* Normal Attention: $O(n^2)$
* LSH Attention: $O(n \log n)$ (kyunki sorting + bucket lookup hota hai instead of all-pairs)

👉 Matlab Reformer **very long sequences (like 64k tokens, 1M tokens)** efficiently handle kar leta hai.

---

## 🔹 Intuition with Example

Socho tumhare paas ek kitab ke 10 lakh words hain:

* Normal Transformer: Har word ko har word se compare karega → \~10¹² operations 😵
* Reformer: Har word ko ek hash bucket me daalega, aur usi bucket ke andar compare karega → bohot kam operations ✅

---

## 🔹 Summary

* **Why LSH?** → To find "approximate nearest neighbors" efficiently.
* **Why Masking?** → Taaki token khud ko na attend kare.
* **Why Multiple Hashes?** → Randomness ko reduce karne ke liye.

---

⚡ Tumhare sentence me jo likha hai:

> "only the biggest elements of QK^T are useful"
> Bilkul sahi — aur LSH basically unhi ko **efficiently approximate** kar raha hai.

---

### **“query = key” case me kya hota hai**.

---

## 🔹 Attention Formula Recap

Self-attention me har token ek **Query (Q)**, **Key (K)**, aur **Value (V)** banata hai:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \text{mask}\right)V
$$

* $Q$ = “What am I looking for?”
* $K$ = “What information I contain?”
* $V$ = “The actual info I can give.”

---

## 🔹 Case: Query = Key (Same Token)

Jab ek token apne aap se compare hota hai (i.e., $q_i \cdot k_i$):

* Dot product hamesha **high value** hota hai, kyunki vector apne hi saath sabse zyada similar hota hai.
* Softmax ke baad us position pe **sabse zyada probability aa jaati hai**.

👉 Result: Token khud pe hi over-focus karne lagta hai, instead of context.

---

## 🔹 Why is this a problem?

1. **Trivial self-copying**
   Model apna hi info baar-baar dekh ke kuch naya learn nahi karega.
   Example: Word "apple" ko process karte waqt, agar wo bas khud hi pe dhyaan de, to context "red", "fruit", "sweet" ignore ho jayega.

2. **Bias in Sparse/LSH Attention (Reformer)**
   Reformer me LSH hashing ke through similar queries aur keys group hote hain.

   * Agar query = key allow kar diya → har token apne hi bucket me aayega → hashing ka fayda kam ho jaata hai.
   * Isliye explicitly **mask kar dete hain self-attention diagonal** (except kuch cases like first token).

---

## 🔹 When is Query = Key Useful?

* **CLS Token (BERT-like encoders)**:
  CLS apne aap pe bhi dhyaan deta hai, kyunki wo aggregate token hai.
* **Diagonal unmasked (GPT)**:
  Decoder me query = key allowed hai (causal mask only blocks *future*), kyunki token apne khud ko bhi use kar sakta hai.

---

## 🔹 Simple Analogy

Socho tum class me ho:

* Agar tum bas **apne khud ke notes** dekhte raho (query = key), tumhe naya kuch nahi milega.
* Agar tum **doston ke notes** bhi dekhte ho (query ≠ key), tumhe broader context samajh aayega.
* Kabhi kabhi apne notes dekhna zaroori hai (useful self-info), but sirf wahi pe focus karna galat hai.

---

👉 **Summary**:

* Query = Key → token apne khud ko hi attend karega (high similarity).
* Encoder models usually allow but balance karte hain.
* Reformer jaise sparse models isko mask kar dete hain (varna trivial attention hogi).
* Decoder (auto-regressive) models me bhi diagonal allowed hota hai (self + past).

---



