
# STM32N6 ONNX Fix Prompt (Complete Repair Instructions for LLM)

## 🚀 Purpose
This prompt is designed for **GPT‑5 / Claude / Gemini / LLMs** to automatically repair an ONNX model so it becomes **fully compatible with STM32 Edge AI Core v2.2.0 (STM32N6)**.

You can paste this entire document directly into an LLM.  
It describes:  
- STM32 official NPU limitations  
- Why the current ONNX model fails  
- What must be fixed  
- What the LLM must generate (Python repair scripts + export pipeline)

---

# ✅ **LLM Prompt: Fix My ONNX Model for STM32N6**

I have an ONNX model that fails on STM32 Edge AI Core v2.2.0 using:

```
stedgeai analyze --model model.onnx --target stm32n6
```

The error is always:

```
INTERNAL ERROR: 'NoneType' object has no attribute 'get_value'
```

STM32 engineers explained that my model violates **official STM32N6 NPU constraints**.

Below is the detailed summary of limitations + the exact issues in my model + what I need you (the LLM) to generate.

---

# 🟥 **A. STM32N6 Official ONNX & NPU Constraints (Must Follow)**

1. **Maximum tensor rank: 5D. No 6D tensor anywhere in ONNX graph.**
2. **Batch size must always be 1. No batched convolution.**
3. **ONNX input must be static shape. No dynamic dimensions.**
4. **Graph must not contain a reshape from 6D → 4D.**
5. **Squeeze must use static `axes=[...]` attribute, not tensor input.**
6. **No dynamic reshape nodes.**
7. **Recommended opset: 14.**

---

# 🟥 **B. My Model Violations (You Must Fix These)**

My original model takes 6D tensors:

```
(B, 8, 3, 36, 36, 3)
```

Violations:

- Model contains internal 6D tensors.
- Contains 6D→4D reshape at the beginning.
- First conv layers operate on input interpreted as batch>1.
- Squeeze nodes use axes from tensor input.
- Some dimensions are still dynamic.

---

# 🟦 **C. What I Need from You**

Generate a **full STM32N6‑compatible repair solution**:

## ✔ 1. Provide corrected model design:
- Preprocessing converts 6D → (1,72,36,36) BEFORE ONNX export.
- ONNX input must be 4D: `(1,72,36,36)`.
- All 6D tensors removed.
- All Squeeze nodes rewritten with static axes.
- No dynamic shapes.

## ✔ 2. Provide Python scripts (onnx + onnx-graphsurgeon) to:
- Remove all 6D reshape nodes.
- Remove/replace 6D Constants.
- Convert Squeeze ops to use static axes.
- Fix all dynamic dims.
- Save repaired model as `model_fixed.onnx`.

## ✔ 3. Provide PyTorch → ONNX export script:
- Input = (1,72,36,36)
- opset_version=14
- dynamic_axes=None

## ✔ 4. Ensure final ONNX passes:
```
stedgeai analyze --model model_fixed.onnx --target stm32n6
```

---

# 🟩 **D. Output Format Required**

1. STM32N6 compatibility checklist  
2. Detailed diagnosis of ONNX issues  
3. Corrected STM32‑friendly model pipeline  
4. Full ONNX repair scripts  
5. Full PyTorch export script  
6. Optional: STM32‑optimized rPPG CNN architecture  

---

# End of Prompt
