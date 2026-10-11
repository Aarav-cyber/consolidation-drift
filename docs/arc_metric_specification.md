# Attractor Regression Coefficient (ARC): Metric Specification

**Project:** Consolidation Drift  
**Author:** Research Team  
**Scope:** Phase 2 Evaluation Metric  

---

## 1. Executive Summary & Problem Formulation

Standard evaluations of long-running LLM agent memory measure either retention ("did the token survive?") or mean signed semantic drift ($\delta_k = \frac{1}{N} \sum_{i} (y_i - x_i)$). 

### The Cancellation Blind Spot of Mean Signed Drift
When evaluating a calibrated corpus spanning both sides of a scale midpoint (e.g., ground truth levels $1, 2$ below center and $4, 5$ above center):
* If repeated consolidation exerts a systematic pull toward a central default prior (e.g., scale level 3):
  * Items starting at level 2 drift up to 3: $\Delta = +1$
  * Items starting at level 4 drift down to 3: $\Delta = -1$
* **Resulting Mean Signed Drift:** $\delta = \frac{(+1) + (-1)}{2} = 0.00$.

Under mean signed drift, such a system appears **completely unbiased** ($\mathcal{H}_0$-consistent). In reality, every single item was distorted toward a central attractor ($\mathcal{H}_1$-consistent). Mean signed drift is structurally blind to symmetric attractor dynamics.

---

## 2. Mathematical Definition of ARC

For any slice of graded memory records (defined by pipeline, context level, and either round $k$ or pooled across rounds):

Let each item $i \in \{1, \dots, n\}$ contribute:
$$x_i = \text{true\_level}_i - 3$$
$$y_i = \text{level}_i - \text{true\_level}_i$$

where:
* $x_i \in [-2, +2]$ is the distance of the item's ground-truth level from the scale midpoint (anchor level 3).
* $y_i \in [-4, +4]$ is the observed signed drift of the evaluated memory.

### Ordinary Least Squares (OLS) Formulation
We fit the linear model:
$$y_i = a + b \cdot x_i + \epsilon_i$$

The **Attractor Regression Coefficient (ARC)** is the slope $b$:
$$b = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2} = \frac{SS_{xy}}{SS_{xx}}$$

with intercept $a = \bar{y} - b \bar{x}$.

### Statistical Inference & Diagnostics
1. **Residual Sum of Squares:**
   $$SS_{\text{res}} = \sum_{i=1}^n \left(y_i - (a + b x_i)\right)^2$$
2. **Coefficient of Determination ($R^2$):**
   $$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{yy}}, \quad \text{where } SS_{yy} = \sum_{i=1}^n (y_i - \bar{y})^2$$
3. **Standard Error of the Slope ($SE(b)$):**
   $$SE(b) = \sqrt{\frac{SS_{\text{res}} / (n - 2)}{SS_{xx}}}$$
4. **$t$-statistic & Two-Tailed $p$-value:**
   $$t = \frac{b}{SE(b)}, \quad p = 2 \cdot P(T_{n-2} > |t|)$$

---

## 3. Physical Interpretation of ARC Values

| ARC Slope ($b$) | Physical Meaning | Theoretical Implication |
| :--- | :--- | :--- |
| **$b \approx 0.0$** | **Position-independent drift.** Movement does not correlate with starting position. Any error is random noise or pure diffusion. | Consistent with **$\mathcal{H}_0$ (Loss/Diffusion)**. |
| **$-1.0 < b < 0.0$** | **Central Attractor Pull.** Items above center drift downward; items below center drift upward. The magnitude $|b|$ indicates the strength of the pull toward the rewriter's unconditioned prior. | Direct signature of **$\mathcal{H}_1$ (Attractor Convergence)**. |
| **$b \approx -1.0$** | **Complete Value Collapse.** All items land at or near the center level ($y_i \approx -x_i \implies \text{level}_i \approx 3$) regardless of initial ground truth. The consolidator has essentially substituted a fixed default statement. | Extreme **$\mathcal{H}_1$ Failure / Information Washout**. |
| **$b > 0.0$** | **Polarization / Sharpening.** Items above center are pushed higher; items below center are pushed lower toward extreme bounds (1 and 5). | **Amplification Bias** (hyperbole/absolutism). |

---

## 4. Empirical Validation on Pilot Data ($n=120$ per cell)

ARC was computed across the 720 graded pilot records from `data/experiments/phase2/evaluations.json`:

| Pipeline | Context Level | Sample Size ($n$) | Mean Signed Drift | Mean Abs Drift | ARC Slope ($b$) | $SE(b)$ | $R^2$ | $p$-value | Interpretation |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`sliding`** | `c0` | 120 | $+0.017$ | $0.017$ | **$+0.0000$** | $0.0144$ | $0.000$ | $1.0000$ | **Inert Control Sanity Check:** Zero LLM calls yields exactly 0 slope. |
| **`sliding`** | `c8` | 120 | $+0.125$ | $0.125$ | **$+0.1250$** | $0.0355$ | $0.095$ | $0.0006$ | Statistically significant mild polarization / sharpening under light context. |
| **`sliding`** | `c20` | 120 | $-0.275$ | $0.275$ | **$-0.1625$** | $0.0666$ | $0.048$ | $0.0161$ | Significant moderate attractor pull under heavy context ($p < 0.05$). |
| **`hierarchical`** | `c0` | 120 | $+0.192$ | $0.375$ | **$-0.2000$** | $0.0752$ | $0.056$ | $0.0089$ | **Key Finding:** Mean signed drift suggested inflation ($+0.192$), but ARC reveals a significant negative attractor ($b = -0.200, p < 0.01$). |
| **`hierarchical`** | `c8` | 120 | $+0.017$ | $0.333$ | **$-0.1375$** | $0.0826$ | $0.023$ | $0.0988$ | Marginal central pull under 8 filler turns. |
| **`hierarchical`** | `c20` | 120 | $-1.100$ | $1.383$ | **$-1.0125$** | $0.1083$ | $0.425$ | $< 10^{-12}$ | Extreme collapse ($b \approx -1.0, R^2 = 0.425, t = -9.35$), reflecting target dilution / fusion washout. |

---

## 5. Limitations & Future Refinements

1. **Range Restriction in Pilot:** The pilot used 12 items spanning true levels $\{2, 3, 4\}$, yielding $x_i \in \{-1, 0, +1\}$. Once scaled to the full 200 items in `atomic_v1.jsonl`, $x_i$ will span $\{-2, -1, 0, +1, +2\}$, doubling statistical leverage.
2. **Linearity Assumption:** OLS assumes a linear relationship across the scale. For extreme boundary behaviors (e.g., truncation at levels 1 and 5), piecewise or logistic formulations may be compared.
3. **Conditioning on Attribute Survival:** ARC must only be computed over records where the target attribute survives. Missing attributes are gated via `ATTRIBUTE_NOT_PRESENT` and analyzed under the companion **Attrition Rate** metric.
