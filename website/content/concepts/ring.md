# The Ring Substrate

Every UOR computation operates over a ring — specifically the modular integer ring Z/(2^n)Z,
where n is determined by the quantum level. This document explains the ring structure, its
physical motivation, and how it grounds the entire ontology.

## What Is a Ring?

A ring is an algebraic structure with two operations, addition and multiplication, satisfying
familiar laws: addition is commutative and associative, multiplication is associative and
distributes over addition, and there exist additive and multiplicative identities.

In UOR, the foundational ring is the byte ring Z/(2^8)Z — integers modulo 256. Every address
in the universal address space, every content-addressed datum, and every identity proof
ultimately reduces to arithmetic in this ring.

The ring is not merely metaphorical. The `schema:Ring` class captures its algebraic structure
formally: the `bitsWidth` property records the bit-width n, and the `schema:QuantumLevel`
individuals Q0–Q3 index the tower of sub-rings Z/(2^1)Z ⊂ Z/(2^2)Z ⊂ Z/(2^4)Z ⊂ Z/(2^8)Z.

## Quantum Levels Q0–Q3

The ring admits a natural four-level scaling:

- **Q0** — 1-bit ring Z/(2)Z. Boolean arithmetic. The atom of all computation.
- **Q1** — 2-bit ring Z/(4)Z. Four elements. The minimal nontrivial modular structure.
- **Q2** — 4-bit ring Z/(16)Z. Nibble-width. Supports geometric intuitions (four quadrants).
- **Q3** — 8-bit ring Z/(256)Z. Byte-width. The universal content-addressing granularity.

Each level is a `schema:QuantumLevel` individual with `quantumIndex` (0–3), `bitsWidth`
(1, 2, 4, 8), and `cycleSize` (2, 4, 16, 256). The `nextLevel` property chains them:
Q0 → Q1 → Q2 → Q3.

Validity in UOR is scoped per quantum level via `ValidityScopeKind`: a theorem can be
`Universal` (valid at all levels), `ParametricLower` (valid at level ≥ k),
`ParametricRange` (valid between k_min and k_max), or `LevelSpecific` (valid only at
exactly one level).

## The `bitsWidth` Property

The `schema:bitsWidth` data property records how many bits a quantum level occupies.
This is not an annotation — it drives computation. The `op:QuantumLevelBinding` class
links operations to their required quantum level, ensuring that proofs and computations
are always validated at the correct ring scale.

## Content Addressing via the Ring

The universal address space in `u:UniversalAddress` is grounded in byte-ring arithmetic.
An address is a point in Z/(2^8)Z^d for some dimension d. Content-addressed objects are
identified by their position in this space, making the ring the literal substrate of every
UOR identity.

The critical identities in `op:criticalIdentity` — the 378 named algebraic identities —
are theorems over this ring: equations that hold in Z/(2^n)Z for the appropriate quantum
level, certified by the proof individuals in `proof:`.

## Connection to the PRISM Pipeline

The ring substrate flows through all three PRISM stages:

- **Define (Kernel)**: The `schema` namespace defines the ring itself and its quantum levels.
  The `op` namespace defines operations over ring elements.
- **Resolve (Bridge)**: Resolvers in the `resolver` namespace compute in the ring.
  The `partition` namespace decomposes the ring into fibers. Homological namespaces
  study ring-level algebraic invariants.
- **Certify (cert)**: Certificates in `cert` attest to ring-level identities holding at
  specific quantum levels, producing the completeness and geodesic certificates that
  close the pipeline.

The ring is the ground on which the entire UOR ontology stands.
