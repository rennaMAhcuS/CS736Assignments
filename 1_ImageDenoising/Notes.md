# Gradient Descent Guidelines

Please stick to classic gradient descent (with dynamic step size) for all your needs:

- Do not use methods involving second derivatives or more complicated schemes.
- For non-convex problems, do not assume any method leads to the global optimum.
- Tune step sizes dynamically; too small is inefficient, too large worsens the objective.

Train all free parameters carefully for denoising tasks (dictionary learning, gradient
descent parameters, etc.). Poor parameter tuning may yield subpar results and lead to
point deductions.

---

## Implementation Steps

1. Scale the image to [0,1] before all processing.
2. Keep your code efficient; a pure CPU implementation can still run in a few minutes.
3. Make parameters for dictionary learning and reconstruction easily adjustable.

Ensure that changes to reconstruction parameters do not affect dictionary-learning
parameters.

---

## Presentations

Each team will give two back-to-back presentations (15 minutes each):

- First covers Q1, Q2, and Q3.
- Second covers Q4.

Evuri Mohana Sreedhara Reddy 23B1017, Gautam Siddharth K 23B0957 – 17:15
