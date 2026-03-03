# Differential Calculus

## Definition

The **discrete differential calculus** of UOR defines two derivative operators
on functions f : R_n → R_n:

- **Ring derivative** ∂_R f(x) = f(succ(x)) - f(x), measuring change along
  the ring successor.
- **Hamming derivative** ∂_H f(x) = f(bnot(x)) - f(x), measuring change
  along the Hamming antipode.

These are encoded as {@class https://uor.foundation/op/Identity} individuals
DC_1 and DC_2 in the `op/` namespace.

## The Jacobian

The {@class https://uor.foundation/observable/Jacobian} decomposes the
incompatibility metric fiber by fiber. At position k:

> J_k(x) = |d_R(x, succ(x)) - d_H(x, succ(x))| restricted to fiber k

Key identities:
- **DC_6**: J_k(x) = ∂_R fiber_k(x)
- **DC_8**: rank(J(x)) = d_H(x, succ(x)) - 1 for generic x
- **DC_9**: Total curvature κ(x) = Σ_k J_k(x)
- **DC_11**: Curvature equipartition — each fiber contributes approximately
  equally to total curvature.

## Curvature-Weighted Resolution

Identity DC_10 shows that the optimal next constraint in iterative resolution
maximizes the Jacobian over free fibers. This connects the differential
calculus to the resolution pipeline: curvature guides constraint selection.

## Commutator Decomposition

Identity DC_4 shows that the fundamental commutator [neg, bnot](x) = 2 can
be recovered from the difference of ring and Hamming derivatives of negation.
This provides a differential-geometric interpretation of the critical identity.
