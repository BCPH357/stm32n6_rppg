
# STM32N6 Edge AI Compatibility & Model Conversion Prompt (Markdown)



# ✅ LLM Prompt: STM32N6 Official Constraints + Error Breakdown + Required Fixes

I have an ONNX model that fails when analyzed with STM32 Edge AI Core v2.2.0 (stm32n6 target).  
The error is always:

```
INTERNAL ERROR: 'NoneType' object has no attribute 'get_value'
```

STM32 engineers replied with the following key points about the model:

---

## 🟥 STM32N6 OFFICIAL LIMITATIONS

1. **STM32 Edge AI NPU only supports tensors up to rank 5.  
   Any 6D tensor anywhere in the ONNX graph is invalid.**

2. My current ONNX graph contains a reshape that reduces **6D → 4D**, meaning the graph still contains 6D tensors internally.  
   Even if the ONNX *input* tensor is 4D, the graph still contains a 6D intermediate tensor.

3. **The first three convolutions operate on a tensor interpreted as “batched” (batch > 1).**  
   This is because the model packs 8 temporal frames × 3×3 channels into a single (72,36,36) input.  
   STM32 compiler interprets this as *multiple images batched together*, which is unsupported.

4. STM32 Edge AI does **NOT** support:
   - Dynamic batch dimensions  
   - Dynamic reshape  
   - Squeeze nodes using axes as a second input tensor  
   - Temporal stacking *inside* the ONNX graph  
   - Any Conv where batch > 1  
   - Models requiring “parallel convolutions sharing weights”

5. After a later reshape, the batch becomes 1, so the remainder of the model is fine —  
   **but STM32 cannot accept any Conv where input batch > 1 even temporarily.**

---

## 🟥 What I Need You (the LLM) To Do

Please generate a **complete STM32N6‑compatible solution**, including:

---

# 🟩 (A) List all STM32N6 ONNX Constraints  
(Max tensor rank, batch rules, dynamic shape rules, unsupported ops, etc.)

# 🟩 (B) Diagnose Exactly How My Model Violates These Rules  
Use my notes + STM32 engineer’s comments:

- 6D→4D reshape still present  
- ONNX contains a 6D Constant  
- First convolutions treated as batched  
- Squeeze op using axes tensor  
- Dynamic shapes causing `'NoneType'.get_value'`  
- Temporal stacking done inside the model

# 🟩 (C) Provide a Correct & STM32-Friendly rPPG Model Design

The corrected pipeline must follow:

### ✔ Preprocessing handles 6D → 4D  
### ✔ ONNX input must be **(1,72,36,36)**  
### ✔ Graph must contain **no** 6D → 4D reshape  
### ✔ No batched convolution  
### ✔ No dynamic dims  
### ✔ All Squeeze must be static attribute `axes=[i]`  
### ✔ opset_version = 14  
### ✔ dynamic_axes = None

---

# 🟩 (D) Provide Complete Python Scripts to Fix/Export the ONNX

Scripts must:

1. Convert (8,3,36,36,3) → (1,72,36,36) **before** ONNX export  
2. Ensure model forward() only accepts 4D input  
3. Remove all 6D→4D reshape nodes  
4. Convert all Squeeze ops to use static axes  
5. Fix all dynamic shapes  
6. Export a final ONNX that passes:

```
stedgeai analyze --model fixed.onnx --target stm32n6
```

---

# 🟩 (E) Optional  
Provide a redesigned rPPG CNN architecture that is:

- NPU-friendly  
- 100% STM32-compatible  
- Uses pure 4D tensors only  
- Avoids any temporal stacking inside the model

---

# 📌 Output Format Required

Please output the following sections:

1. **STM32N6 compatibility checklist**  
2. **Detailed diagnosis: how my model violates each rule**  
3. **Corrected architecture & pipeline**  
4. **Full working Python scripts** for:
   - ONNX graph cleanup  
   - Squeeze fixing  
   - Static reshape injection  
   - ONNX export  
5. **Final notes & best practices** for STM32 Edge AI model construction

---

# End of Prompt

