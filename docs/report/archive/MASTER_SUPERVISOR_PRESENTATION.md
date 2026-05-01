# Set-ConCA: Hybrid Master Presentation for Supervisor
**Goal:** 15-Minute Project Review. Combines NeurIPS-grade results with candid internal reflections.

---

## Slide Storyboard & Narrative

### Slide 1: The Set-ConCA Objective
*   **Visuals:** Comparison of "Pointwise" vs "Set" encoding.
*   **Narrative:** "We set out to fix the 'Surface-Level Noise' in Sparse Autoencoders. Most SAEs overfit to specific word choices. By training on **paraphrase sets**, we forced the model to extract the invariant semantic core. This is a report on that 13-experiment journey."

### Slide 2: Structural Innovation (The Dual Decoder)
*   **Visuals:** Architecture diagram highlighting the Shared vs Residual streams.
*   **Narrative:** "We didn't just pool the data; we built a **Dual Decoder**. This formally separates 'Concepts' from 'Context.' It allowed us to reduce reconstruction error by **45%** compared to naive set-averaging. This is the foundational block of our system."

### Slide 3: Validating the "Platonic" Hypothesis
*   **Visuals:** Gemma-4B $\to$ LLaMA-8B Transfer bar chart (**64.6%**).
*   **Narrative:** "Highest-value result: Concepts transfer across model families. Even though Gemma and Llama were trained by different companies on different data, they converge to the same semantic geometry. Set-ConCA is the 'bridge' that reveals this shared reality."

### Slide 4: Honest Reflection: The "Over-Engineering" Trap
*   **Visuals:** Diagram showing Consistency Loss vs TopK Bottleneck.
*   **Narrative:** "**Candid takeaway:** We initially built a complex 'Consistency Loss' to force stability. **Tests (EXP9) proved it was redundant.** The TopK sparsity bottleneck is so strong that it does the stabilization for free. In future projects, we can strip this out to save **15% in training time.**"

### Slide 5: The "Weak-to-Strong" Steering Surprise
*   **Visuals:** 1B model steering an 8B model curve.
*   **Narrative:** "We found a 'Golden Path' for safety research. A small 1B-model concept vector steers an 8B model **better** than a 4B one does. This proves that aggressive compression actually 'distills' the most causal directions. We can use cheap models to control expensive ones."

### Slide 6: Linear vs Nonlinear: The Geometry Test
*   **Visuals:** Comparison of a simple Rotation (Linear) vs a 2-layer MLP bridge.
*   **Narrative:** "We checked if model-to-model alignment was complex. **It's not.** The MLP bridge only added **0.5pp** in accuracy. This is a major win for simplicity—it means the concept spaces are **isometric** (rotated versions of each other)."

### Slide 7: Robustness & The Noise Ceiling
*   **Visuals:** Transfer accuracy vs Noise level.
*   **Narrative:** "We tried to break the model (EXP10) by feeding it 100% noise paraphrases. It didn't break. The system is incredibly robust to 'dirty' training data. This means we don't need perfect human-labeled paraphrases; noisy LLM outputs are enough."

### Slide 8: Dimensional Depth: Where the Meaning Lives
*   **Visuals:** PCA Rank vs Transfer peak at Rank 32.
*   **Narrative:** "Where is the 'Human Meaning' in a 3000-dimensional vector? We found it’s mostly in the **first 32 dimensions**. Beyond that, it’s model-specific 'syntactic fluff.' This allows us to massively compress our SAEs without losing steering capability."

### Slide 9: Economic & Strategic Value (The "So What?")
*   **Visuals:** List of Reusable Assets.
*   **Narrative:** "Beyond the paper, we now have: 
    1. A **Universal Concept Porting** pipeline.
    2. A verified **Weak-to-Strong** steering methodology. 
    3. A much **cleaner training objective** (now that we know which losses to cut)."

### Slide 10: Conclusion & Next Steps
*   **Narrative:** "Set-ConCA isn't just a research paper; it's a blueprint for **cross-model interpretability**. I'm ready to move into the full-scale Llama-70B validation using the 1B 'Probe' methodology we validated here."

---

## The Supervisor Q&A "Defense" (Candid Edition)

| Question | The Academic Answer | The Internal "Real" Answer |
| :--- | :--- | :--- |
| **Why not just use Pointwise SAEs?** | "They fail to align across families." | "They overfit to syntax. You get 8% worse alignment." |
| **How do we speed up training?** | "Optimization of hyperparameters." | "Delete the Consistency Loss. It's doing nothing." |
| **Is 64% transfer good enough?** | "It beats baselines significantly." | "It's the ceiling for mid-layers. Higher layers likely need a sparse-map." |
| **Will this work on GPT-4?** | "If we get activations, the theory holds." | "The linear bridge theory says yes. Geometry is convergent." |

---
*For raw data, see: [REPORT_DENSE.md](file:///c:/Users/MPC/Documents/code/SetConCA/results/REPORT_DENSE.md)*
*For technical logs, see: [SUPERVISOR_BRIEFING.md](file:///c:/Users/MPC/Documents/code/SetConCA/results/SUPERVISOR_BRIEFING.md)*
