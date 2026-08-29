# MINRES: In Between GMRES and CG

**Author:** D. Carter

This repository contains a from-scratch derivation and implementation of the **MINRES algorithm** for solving symmetric linear systems $Ax = b$, built up from the Arnoldi process through to a Givens-rotation-based short-recurrence update.

---

## Overview

**MINRES** is a Krylov subspace method for symmetric linear systems $Ax = b$. At each iteration $k$, it finds the vector $x_k$ in the $k$-dimensional Krylov subspace that minimizes the residual $\|Ax_k - b\|_2$.

This project derives MINRES starting from the more general **Arnoldi process / GMRES**, and shows step by step how:

- Symmetry of $A$ collapses the Arnoldi relation into a **three-term Lanczos recurrence**.
- The MINRES subproblem reduces to a **least-squares problem with a tridiagonal matrix**.
- That tridiagonal system's **QR factorization via Givens rotations** can be computed in $O(n)$, versus $O(n^3)$ in general.
- The QR factors give a **short-recurrence update rule** for $x_k$, avoiding the need to store the full Krylov basis (unlike GMRES).

The notebook derives, implements, and empirically tests this algorithm, then compares its cost and convergence behavior against CG and GMRES.

---

## Derivation & Implementation Details

- **Derivation path:** Arnoldi process → GMRES formulation → Lanczos three-term recurrence (symmetric case) → tridiagonal least-squares subproblem → Givens QR → short-recurrence update for $x_k$.
- **QR via Givens rotations:** exploits the tridiagonal structure to compute/update the QR factorization in $O(n)$ per step, instead of the general $O(n^3)$.
- **Implementation:** Lanczos iteration + Givens QR + short-recurrence update, implemented in NumPy.
- **Experiments:** convergence of MINRES tested against matrices with varying condition number and eigenvalue clustering, checked against the known convergence bound; memory and computational cost compared against CG and GMRES.

### Note on the implementation

The `MINRES` function follows the derivation exactly but has an unresolved numerical bug that wasn't fixed in time. The convergence experiments use `scipy.sparse.linalg.minres` instead, to validate the theoretical results independently of that bug.

---

## Key Theoretical Results

### Reduction to a tridiagonal least-squares problem

For symmetric $A$, the Arnoldi relation $AQ_k = Q_{k+1}\tilde H_k$ simplifies to $AQ_k = Q_{k+1}T_k$ with $T_k$ tridiagonal, giving the MINRES subproblem at each step:

$$
y_k = \arg\min_{y \in \mathbb{R}^k} \|T_k y - \|b\|_2 e_1\|_2
$$

### Efficient QR of a tridiagonal matrix

Using Givens rotations to zero the single subdiagonal at each step, the QR factorization $T = Q^*R$ of an $n \times n$ tridiagonal matrix can be computed in $O(n)$, compared to $O(n^3)$ for a general QR decomposition.

### Short-recurrence update

Combining the QR factors with the least-squares solution gives an update rule requiring only the last two auxiliary vectors:

$$
p_k = \frac{1}{r_{kk}}\left(q_k - p_{k-2}r_{k-2,k} - p_{k-1}r_{k-1,k}\right), \qquad x_k = x_{k-1} + \|b\|_2\, p_k\, q_{1,k}^{*}
$$

This is what makes MINRES memory-efficient relative to GMRES, which must store the entire Krylov basis.

---

## Convergence Experiments: Condition Number and Eigenvalue Clustering

MINRES's convergence rate is governed by the bound

$$
\frac{\|b - Ax_k\|_2}{\|b - Ax_0\|_2} \le \min_{p \in \mathcal{P}_k} \max_{\lambda \in \Lambda(A)} |p(\lambda)|
$$

This bound isolates two independent effects: how *spread out* the eigenvalues are (condition number) and how *clustered* they are. Three experiments were run to separate these effects:

| Experiment | Setup | Result |
|---|---|---|
| Same condition number, different clustering | Two matrices with identical condition number $\kappa(A)$, one with evenly spaced eigenvalues, one with tightly clustered eigenvalues | The clustered-eigenvalue matrix converged in noticeably fewer iterations. |
| Different condition number, different clustering | $\kappa(A)$ and clustering varied together | Convergence tracked clustering more than condition number alone. |
| Different condition number, same clustering | Evenly spaced eigenvalues at two different scales (same relative spacing) | Convergence speed was comparable, isolating clustering as the dominant factor. |

**Conclusion:** the more tightly clustered the eigenvalues of $A$, the fewer iterations MINRES needs to converge — consistent with the theoretical bound above, since a low-degree polynomial can be made small across a tight cluster of eigenvalues far more easily than across a widely spread set.

---
