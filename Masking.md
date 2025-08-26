## 🔹 What is Masking in Transformers?

Masking ka matlab hota hai **kuch positions ko ignore karna** (ya "chhupana") jab attention compute hota hai.

* Jab hum $QK^T$ nikalte hain, usme har query word ko har key word se similarity milti hai.
* Kabhi kabhi hume **sab connections allow nahi karne hote**, isliye ek **mask matrix** apply kar dete hain.
* Masked positions ka weight **-∞** (ya bahut bada negative number) daal dete hain → softmax ke baad wo effectively **0 ban jaata hai**.

---

## 🔹 Types of Masking

1. **Padding Mask**

   * Sentences alag-alag length ke hote hain. Batch banate waqt short sentences ko `PAD` tokens se fill karna padta hai.
   * Ye `PAD` tokens **meaningless** hote hain, to model ko unpe dhyaan nahi dena chahiye.
   * Example:

     ```
     Sentence 1: I love cats
     Sentence 2: Dogs are
     After padding: Dogs are [PAD]
     ```

     Attention mask ensure karega ki `[PAD]` tokens ignore ho.

---

2. **Causal (Look-ahead) Mask**

   * Decoder models (GPT jaise) me **future tokens nahi dekhne chahiye**.
   * Jab word `t` predict ho raha hai, wo sirf positions `[0..t]` tak access kare.
   * Ye ek **triangular mask** hota hai jo future ko block kar deta hai.
   * Example:

     ```
     "I love cats"
     Word "cats" ko predict karte waqt "cats" khud aur uske baad ke words ko hide kar dete hain.
     ```

---

3. **Special Masking (Reformer / LSH / Sparse Attention)**

   * Yaha masking use hota hai **self-attention ke unwanted parts** ko block karne ke liye.
   * Example: Reformer me **q = k** (same token) case ko mask karte hain, warna token khud pe hi high attention dega.

---

## 🔹 Intuition

Socho ek classroom hai:

* **Padding mask** = khaali kursi (student hi nahi hai) → teacher ignore karega.
* **Causal mask** = student ko sirf usse pehle bolne wale students sunne ki permission hai, baad ke nahi.
* **Reformer mask** = student apne hi khud ke answer ko nahi dekh sakta, warna cheating ho jaayegi. 😅

---

## 🔹 Math Form

Attention score:

$$
\text{scores} = \frac{QK^T}{\sqrt{d_k}} + \text{mask}
$$

* Mask me unwanted jagah pe **-∞** daal do.
* Softmax ke baad:

  $$\text{softmax}(-∞) = 0$$

  Matlab woh connection ignore ho gaya.

---

👉 Matlab: **Masking = controlling "who can attend to whom".**

---

