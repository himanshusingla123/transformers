**Reformer** uses to handle **very long sequences** efficiently. Let’s break down **axial positional encodings** in simple terms:

---

### 🔹 Background: Why Positional Encodings?

Transformers don’t have a sense of order by default (they just see tokens as a set).
So we add **positional encodings (PEs)** → vectors that tell the model **“this token is at position 1, this token at position 2, etc.”**

Normally:

* If sequence length = `l`
* Hidden dimension = `d`
* PE matrix size = `l × d`

👉 For long sequences (say `l = 1,000,000`, `d = 1024`), that’s **1 billion values** → too big for GPU.

---

### 🔹 Reformer’s Solution: Axial Positional Encodings

Instead of storing a **huge matrix**, Reformer **factorizes it into two smaller ones**.

We split:

* Sequence length → `l = l₁ × l₂`
* Hidden dimension → `d = d₁ + d₂`

Now we build:

* `E₁` → size `(l₁ × d₁)`
* `E₂` → size `(l₂ × d₂)`

The position `j` in the sequence is encoded as:

```
PE[j] = concat( E₁[j % l₁], E₂[j // l₁] )
```

---

### 🔹 Example

Suppose:

* Sequence length `l = 12`
* Hidden dim `d = 6`

We choose:

* `l₁ = 4`, `l₂ = 3`  (since `4 × 3 = 12`)
* `d₁ = 3`, `d₂ = 3`

So:

* `E₁` is a `4 × 3` matrix
* `E₂` is a `3 × 3` matrix

Now:

* Position `j = 7` →

  * `j % l₁ = 7 % 4 = 3` → take row 3 of `E₁`
  * `j // l₁ = 7 // 4 = 1` → take row 1 of `E₂`
  * Final encoding = **concat(row3(E₁), row1(E₂))**

---

### 🔹 Why is this useful?

* Memory is much smaller (`l₁·d₁ + l₂·d₂` instead of `l·d`).
* Still uniquely encodes positions across very long sequences.
* Works especially well with **LSH attention** (Reformer’s other trick).

---

✅ In short: **Axial PEs = breaking a huge position matrix into 2D factors**, like reshaping a 1D sequence into a grid, and then encoding row + column instead of each position separately.
