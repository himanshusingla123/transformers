# 🔹 Local Attention (Sliding Window)

* Normal self-attention me **har token har dusre token ko dekh sakta hai** → complexity $O(n^2)$.
* Longformer isko optimize karta hai → **local sliding window** use karke:

  * Har token sirf apne **nearby neighbors** (e.g. left/right ke 2-3 tokens) ko attend karega.
  * Example: Sentence = "The quick brown fox jumps"

    * Token "brown" → sirf \["quick", "brown", "fox"] pe dhyaan de.
  * Complexity reduce ho jaati hai: $O(n \cdot w)$, jaha $w$ = window size (bahut chhota compared to $n$).

👉 **Benefit**: Efficient, suitable for **long sequences** (like documents, research papers).

---

# 🔹 Receptive Field Expansion (Stacked Layers)

* Agar ek layer ka window size = 3 hai, to us layer ka token max 3 neighbors ko dekhta hai.
* Lekin agar tum 12 layers stack karte ho →

  * Layer 1: dekhta hai **3 tokens**
  * Layer 2: indirectly unke aur 3 tokens ko
  * … aur aise hi expand hote hote end tak **poora sentence ka context build ho jata hai**.

👉 Yaani **small local window se bhi global info banayi ja sakti hai (deep stacking se)**.

---

# 🔹 Global Attention (Selected Tokens)

* Problem: Local attention sirf chhote window tak hi limited hai. Kuch tokens ko **global info chahiye** (jaise question tokens in QA).
* Isliye Longformer kuch **special tokens** ko **global attention** assign karta hai:

  * Ye tokens har ek token ko dekh sakte hain.
  * Aur sabhi tokens in global tokens ko dekh sakte hain.
  * Symmetric hai → 2-way communication.

### Example:

* **Q\&A Task**:

  * Question tokens ko **global attention** do → taaki wo poore passage ko attend kar saken.
  * Passage tokens sirf apne local neighbors ko dekhen, but **question ko bhi access kar saken**.

---

# 🔹 Attention Masking in Longformer

Attention mask = ek **matrix** jo batata hai kaunsa token kaunsa attend kar sakta hai.

* Local attention = sirf ek **banded diagonal mask** (jaise ek stripe, ±window ke andar).
* Global attention = poore row/column un tokens ke liye 1 (universal access).

---

# 🔹 Visual (Simplified Mask)

```
Local Attention (window=2):

    T1 T2 T3 T4 T5
T1  *  *           (T1 dekhe T1,T2)
T2  *  *  *        (T2 dekhe T1,T2,T3)
T3     *  *  *     (T3 dekhe T2,T3,T4)
T4        *  *  *  (T4 dekhe T3,T4,T5)
T5           *  *  (T5 dekhe T4,T5)

Global Token (say T1 is global):

    T1 T2 T3 T4 T5
T1  *  *  *  *  *   (T1 dekhe sabko)
T2  *  *  *        (T2 still local, but +T1)
T3  *     *  *     (T3 still local, but +T1)
T4  *        *  *  (T4 still local, but +T1)
T5  *           *  (T5 still local, but +T1)
```

---

# 🔹 Analogy

Socho tum ek **classroom** me ho:

* Local attention = tum sirf apne aas-paas baithne walon se baat kar sakte ho.
* Global attention = teacher (global token) → wo sabse baat kar sakta hai, aur sab usse baat kar sakte hain.

---

👉 **Summary**:

* Local attention = sliding window → efficient for long texts.
* Global attention = few special tokens → bridge between local groups and global context.
* Is combination se Longformer ban jata hai **scalable + context-aware**.
