# Analytical Completeness

## Definition

**Analytical completeness** means that the UOR ontology provides a complete
topological and spectral characterization of the resolution process. Three
structures make this possible: the constraint nerve, Betti numbers, and the
index theorem.

## Constraint Nerve

The {@class https://uor.foundation/resolver/ConstraintNerve} is the simplicial
complex whose vertices are constraints and where a k-simplex exists iff the
corresponding k+1 constraints have nonempty pin intersection. Identity HA_1
formalizes this construction.

The nerve's topology governs resolution behavior:
- **Trivial homology** (all Betti numbers zero) → smooth convergence
- **Non-trivial homology** → potential stalls (identity HA_2)

## Betti Numbers

{@class https://uor.foundation/observable/BettiNumber} β_k = rank(H_k(N(C)))
counts the k-dimensional holes in the constraint configuration. The
Betti-entropy theorem (HA_3) gives a lower bound on residual entropy:

> S_residual ≥ Σ_k β_k × ln 2

## Spectral Gap

The {@class https://uor.foundation/observable/SpectralGap} λ_1 is the smallest
positive eigenvalue of the constraint nerve Laplacian. Identity IT_6 shows
that λ_1 lower-bounds the convergence rate of iterative resolution.

## The UOR Index Theorem

The capstone identity IT_7a connects curvature, topology, and entropy:

> Σ κ_k - χ(N(C)) = S_residual / ln 2

where κ_k is the total curvature at fiber k, χ is the Euler characteristic,
and S_residual is the residual Shannon entropy. This is the UOR analog of the
Atiyah-Singer index theorem.

Consequences:
- **IT_7b**: S_residual = (Σ κ_k - χ) × ln 2
- **IT_7c**: Resolution cost ≥ n - χ(N(C))
- **IT_7d**: Resolution is complete iff χ(N(C)) = n and all β_k = 0
