# System / Role

You are an expert in:
- remote photoplethysmography (rPPG) modeling,
- PyTorch model design and training,
- ONNX / TFLite export and post-training quantization (QDQ),
- deployment to STM32N6 using ST Edge AI Developer Cloud and X-CUBE-AI 10.2.0 (stedgeai v2.2.0).

Your job is to **help me redesign, retrain, quantize, and deploy** my multi‑ROI rPPG model so that it:
1. Keeps (or improves) accuracy vs my current PyTorch model.
2. Is **fully compatible with STM32N6 NPU** (no internal error in `stedgeai analyze --target stm32n6`).
3. Produces an **INT8 QDQ ONNX/TFLite model** suitable for STM32 Edge AI Core v2.2.0.

I am OK to retrain the model from scratch as long as the architecture still fits my data format and is ultra‑lightweight.

---

## 1. Dataset & Preprocessing (UBFC, Multi‑ROI, 8‑frame window)

I am using the **UBFC‑rPPG DATASET_2** with my own preprocessing pipeline.

Key properties:

- Dataset: UBFC‑rPPG / DATASET_2.
- For each subject:
  - I read `vid.avi` and `ground_truth.txt`.
  - From `ground_truth.txt`:
    - **Line 1**: PPG waveform (BVP).
    - **Line 3**: timestamps in seconds.
- I compute frame‑wise HR using a **robust peak‑based method**:
  - Butterworth bandpass filter on PPG: 0.7–3.0 Hz (≈ 42–180 BPM).
  - `scipy.signal.find_peaks` with constraints on:
    - `distance` (≈ 0.35 s min).
    - `prominence`.
    - `width`.
  - Compute RR intervals, enforce physiologic ranges (0.3–1.5 s).
  - Convert RR to HR (BPM) and apply multiple cleaning stages:
    - Filter HR to [40, 160] BPM.
    - Remove outliers, interpolate HR to every video frame.
    - Final HR per frame is robust and constrained to [40, 160] BPM.
- For each video frame:
  - I detect a face with OpenCV Haar Cascade (`haarcascade_frontalface_default`).
  - I extract **3 ROIs**:
    1. Forehead
    2. Left cheek
    3. Right cheek
  - Each ROI is resized to **36×36**, normalized to [0, 1], RGB.
  - If ROI extraction fails, I use a zero patch.
- I then build temporal windows:
  - Window size `T = 8` frames, stride usually = 1.
  - For each temporal window, I use the **middle frame HR** as label.
  - I filter out windows if:
    - Any HR in the window is out of [40, 160] BPM.
    - HR std in the window is ≥ 15 BPM.
- Final saved data (in `ubfc_processed.pt`):
  - `samples`: tensor of shape **(N, 8, 3, 36, 36, 3)**  
    - N windows, T=8 frames, 3 ROIs, 36×36, 3 channels (RGB), normalized float32.
  - `labels`: tensor of shape **(N, )**, each is HR in BPM (float32, robust peak‑based).

I will provide you the preprocessing script if needed, but you can assume the final training data has shape:

```text
X: (N, 8, 3, 36, 36, 3)
y: (N,)  # HR in BPM, range [40, 160]
```

---

## 2. Current Model Architecture (UltraLightRPPG, PyTorch)

This is my current PyTorch model that I plan to retrain. It is designed as an **ultra‑lightweight multi‑ROI rPPG model**:

- Input: `(B, T, ROI, H, W, C) = (B, 8, 3, 36, 36, 3)`
- Output: `(B, 1)` → HR in BPM, constrained roughly to [30, 180].
- Structure:
  1. **Shared 2D CNN** (`self.spatial`) applied to each ROI frame:
     - 3→16 Conv2d + BN + ReLU + MaxPool (36×36 → 18×18)
     - 16→32 Conv2d + BN + ReLU + MaxPool (18×18 → 9×9)
     - 32→16 Conv2d + BN + ReLU
     - Global Average Pooling to get a **16‑dim feature per ROI per frame**.
  2. For each time window:
     - For each of the 8 frames and 3 ROIs:
       - Apply `self.spatial` (weights shared across ROIs & time).
       - Get spatial features `(B, T, ROI, 16)`.
  3. **ROI fusion**:
     - Concatenate ROI features along ROI axis → `(B, T, 48)` (because 16×3=48).
  4. **Temporal Conv1D** (`self.temporal`):
     - Transpose to `(B, 48, T)`.
     - Conv1d(48→32, kernel=3, padding=1) + ReLU.
     - Conv1d(32→16, kernel=3, padding=1) + ReLU.
     - Output shape: `(B, 16, T)` (T=8).
  5. **Flatten + FC** (`self.fc`):
     - Flatten to `(B, 16 * T) = (B, 128)`.
     - Linear(128→32) + ReLU.
     - Linear(32→1).
  6. **Output activation** (`self.output_act`):
     - `Sigmoid` then scaled: `hr = sigmoid(out) * 150 + 30`, so HR is ≈ [30, 180] BPM.

I will feed the full `UltraLightRPPG` class to you, but you can assume the core logic is exactly as described above.

**Important:**  
This new architecture already avoids the previous 6D→4D reshaping trick and separates spatial & temporal modeling in a cleaner way.

---

## 3. Target Deployment & Toolchain (STM32N6 + Edge AI)

My final target is:

- MCU: **STM32N6** (with NPU).
- Toolchain: **ST Edge AI Core v2.2.0-20266**, via:
  - STM32Cube.AI / X‑CUBE‑AI 10.2.0 (local, Windows).
  - ST Edge AI Developer Cloud (CLI `stedgeai`).
- Commands I use (typical):

```bash
stedgeai analyze --model <model> --target stm32n6 --name network   --workspace <workspace> --output <output>
```

I want to:

1. Export my final rPPG model to **ONNX and/or TFLite**.
2. Run **INT8 QDQ quantization** (my earlier experiments with ONNX QDQ gave excellent results: MAE degradation only ~0.2 BPM).
3. Run `stedgeai analyze --target stm32n6` successfully.
4. Generate C code to run on STM32N6 with the NPU (not just CPU).

---

## 4. Problems I Previously Hit (You Should Avoid)

Before redesigning the model, I had a different architecture that caused a lot of issues with STM32 Edge AI. I want you to be aware of these problems so that you can **intentionally avoid them** in your proposed solution.

### 4.1 6D Input + Reshape → ONNX not supported by ST

Old model input shape:

```text
(B, 8, 3, 36, 36, 3)
```

I used ONNX with a `Reshape` from 6D to 4D:

```text
(B, 8, 3, 36, 36, 3)  →  (B, 72, 36, 36)
```

This caused:

- `INTERNAL ERROR: Unexpected combination of configuration and input shape`
- `INTERNAL ERROR: 'NoneType' object has no attribute 'get_value'`

From ST support (Julian), I got the feedback that:

- **Max supported tensor rank is 5D**, and
- They **do not support batched convolutions** where the first dimension >1 before certain reshapes.
- The first few convolutions using a batch dimension as a pseudo feature dimension is not supported.
- They suggested splitting batched convolutions into explicit parallel convolutions sharing weights or refactoring the model so that:
  - The NPU sees simple 4D / 5D tensors in standard shapes.

This is one key reason I redesigned the architecture to the current `UltraLightRPPG` model with a cleaner temporal Conv1D.

### 4.2 ONNX Identity Models Failing

I also tested very simple ONNX models with:

- Input: `[1, 3, 36, 36]` or `[1, 72, 36, 36]`
- A single `Identity` node (no CNN)

Even these models caused errors like:

```text
INTERNAL ERROR: 'input'
INTERNAL ERROR: 'input_0'
INTERNAL ERROR: Order of dimensions of input cannot be interpreted
```

So I concluded:

- The ONNX route is quite fragile in my environment with ST Edge AI v2.2.0.
- It might be much safer to focus primarily on **TFLite** as the deployment format, or at least design ONNX in a very ST‑friendly way.

### 4.3 TFLite Tests and Channel Constraints

I created test TFLite models using TensorFlow:

1. Minimal model with:
   - Input: `[1, 192, 192, 3]`  
   - Conv2D + Dense
   - Result: `stedgeai analyze --target stm32n6` **works**.

2. Minimal model with:
   - Input: `[1, 36, 36, 3]`  
   - Identity or small Conv2D
   - Result: **works** (analyze OK).

3. Minimal model with:
   - Input: `[1, 36, 36, 72]` (to simulate 72 channels)
   - Identity or simple Conv2D(1×1)
   - Result: `INTERNAL ERROR: 'NoneType' object is not subscriptable`.

From these experiments I infer:

- STM32 Edge AI is **happy with 3‑channel (RGB) images** in H×W format.
- It tends to **reject "weird" high‑channel inputs like 72 channels**, even in TFLite.
- Therefore, the deployment‑friendly design should never push a 72‑channel pseudo‑image into the NPU as the main input tensor.

This also justifies the new `UltraLightRPPG` design, where:

- Each frame is processed as a normal 3‑channel 2D image (36×36×3).
- Temporal modeling happens in a 1D Conv over feature vectors, not by abusing 2D Conv over a 72‑channel image.

---

## 5. What I Want You To Do

Given all the above context, I want you to:

### 5.1 Validate / Improve the UltraLightRPPG Architecture

1. Carefully read and analyze my current `UltraLightRPPG` PyTorch model:
   - Input: `(B, 8, 3, 36, 36, 3)`.
   - Shared 2D CNN per ROI with GAP → (16‑dim per ROI per frame).
   - ROI fusion → 48‑dim per frame.
   - Temporal Conv1D → 16 channels over 8 time steps.
   - FC → HR in BPM.
2. Confirm whether this architecture is:
   - Conceptually sound for rPPG (multi‑ROI, temporal modeling).
   - Friendly for quantization (INT8 QDQ).
   - Friendly for STM32N6 NPU mapping (after proper export/refactoring).
3. Suggest **minor architecture tweaks** if needed to further improve:
   - Robustness.
   - Latency / memory footprint on STM32N6.
   - Edge‑friendliness (e.g., avoiding layers that are slow on NPU).

### 5.2 Design a Deployment‑Friendly Graph for STM32N6

I want you to help me design a **concrete deployment strategy** that satisfies STM32N6 constraints. For example, you can propose one of these patterns:

- **Pattern A (Recommended):**
  - Core model exported as:
    - Input: `[1, 3, 36, 36]` (NCHW) or `[1, 36, 36, 3]` (NHWC), representing 1 ROI frame.
    - Output: 16‑dim feature vector for that ROI frame.
  - On MCU, explicitly loop over:
    - T=8 frames.
    - ROI=3.
  - Fuse features and run a separate small temporal model (Conv1D + FC) either:
    - on CPU (C code), or
    - as another small NN model that consumes `[1, 48, 8]` or `[1, 8, 48]`.

- **Pattern B:**
  - Single NN model with input shape `[1, 8, 3, 36, 36, 3]` **at training time**, but at export time you refactor the graph into:
    - multiple calls with 4D inputs.
    - or explicit reshape ops that stay within 5D and are supported by N6.

Your task:
- Choose a concrete pattern,
- Explain in detail how inputs and outputs will be shaped at deployment,
- Ensure it satisfies:
  - Max 5D tensors,
  - No abusive batched Conv as temporal conv,
  - Reasonable channel sizes (C in {1,3,4,…,small number}).

### 5.3 Provide a Full Training + Export + Quantization Pipeline

I want you to **write code and instructions** that cover the full path:

1. **PyTorch training code** for `UltraLightRPPG` (or your improved variant):
   - Dataloader for `(N, 8, 3, 36, 36, 3)` from `ubfc_processed.pt`.
   - Reasonable loss function (e.g. L1 / SmoothL1 / L2 on HR).
   - Training loop with logging (MAE / RMSE in BPM).
2. **Model evaluation** code on a validation set.
3. **Export to ONNX and/or TFLite**:
   - Make sure to explicitly control:
     - Input shape and layout (NCHW/NHWC).
     - opset version.
   - If ONNX is used:
     - Avoid 6D tensors.
     - Avoid operations known to break STM32 Edge AI.
   - If TFLite is used:
     - Use TensorFlow / Keras wrappers to recreate the model exactly or approximately.
4. **Quantization (INT8 QDQ)**:
   - For ONNX:
     - Show how to use `onnxruntime.quantization` to create a QDQ model.
     - Use a representative calibration dataset drawn from my `ubfc_processed.pt` data.
   - For TFLite:
     - Show how to use post‑training quantization (int8 weights + activations) with a representative dataset.
5. Ensure the exported model(s) can be passed to:

```bash
stedgeai analyze --model <final_model> --target stm32n6
```

without internal errors.

### 5.4 (Optional But Very Helpful) Knowledge Distillation

If you think it helps, you can also propose a **knowledge distillation setup**:

- Teacher: my current best FP32 model (possibly a bigger model I already had).
- Student: the new ultra‑light STM32N6‑friendly model.
- Distillation loss + HR loss combined.

You don’t need to know the exact teacher architecture; you can assume I can plug in its predictions.

---

## 6. Output Style

When you answer, please:

1. Be **very explicit about tensor shapes** at each stage (training vs export vs deployment).
2. Prefer **PyTorch examples** for training and **TensorFlow / TFLite or ONNX** examples for export.
3. Clearly mark **which parts are mandatory for STM32N6 compatibility**.
4. Try to keep the model small (e.g. total params ≲ 100K).

You can assume I am comfortable implementing Python and C code, but I need your help to:

- Design the STM32N6‑friendly model and graph,
- Avoid ST Edge AI Core pitfalls,
- Build a reliable training+deployment pipeline end‑to‑end.
