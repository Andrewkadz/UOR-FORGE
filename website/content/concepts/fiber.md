# Fiber Bundle Semantics

UOR organizes typed data using the mathematical structure of a fiber bundle. Understanding
fiber bundles explains why types in UOR behave the way they do, and why the partition
namespace exists as a structural separator between kernel and user concerns.

## What Is a Fiber Bundle?

A fiber bundle consists of three components:

1. **Total space E**: the full collection of all typed data
2. **Base space B**: the address space — the set of all possible content addresses
3. **Fiber F**: the type (or set of types) that lives over each address

For every address b in B, there is a fiber F_b — a set of valid types at that address.
The total space E is the disjoint union of all fibers: E = ∪_{b∈B} F_b.

In UOR, the base space is the `u:UniversalAddress` space (grounded in the ring Z/(2^8)Z),
and fibers are `type:Fiber` instances — the possible types attached to a given address.

## The `type` Namespace

The `type` namespace is the User-space implementation of fiber bundle semantics.
Key classes:

- `type:Fiber` — the fundamental typed layer over an address
- `type:FlatType` — a fiber with trivial (non-twisted) holonomy
- `type:TwistedType` — a fiber with nontrivial holonomy (monodromy acts non-trivially)
- `type:CompleteType` — a fiber whose completeness has been certified
- `type:TypeSynthesisGoal` and `type:TypeSynthesisResult` — classes for the ψ-pipeline
  inversion: computing what type is needed given a constraint

The `type:amplitude` property records the quantum amplitude of a superposed fiber state,
connecting the fiber bundle structure to the quantum level system.

## The `partition` Namespace

The `partition` namespace implements the decomposition of the base space. An address space
can be partitioned into disjoint subsets (fibers in the partition sense), and UOR uses
this to implement namespace-level modularity.

Key classes:

- `partition:Partition` — a decomposition of the address space
- `partition:PartitionProduct` — the Cartesian product of two partitions (the full space)
- `partition:PartitionCoproduct` — the disjoint union of two partitions (exclusive choice)
- `partition:AncillaFiber` — an auxiliary fiber used in reversible computation strategies

The `partition:isExhaustive` property asserts that a partition covers the entire space —
a formal completeness claim implemented in OWL.

## Connection to Holonomy

When a fiber is transported around a loop in the base space, it may return to a different
element of the fiber — this transformation is the **holonomy**. If all holonomies are trivial
(the identity), the bundle is flat (`type:FlatType`). Otherwise, it is twisted
(`type:TwistedType`).

The `observable:HolonomyGroup`, `observable:Monodromy`, and `observable:MonodromyClass`
classes in the Bridge space observe these holonomy phenomena. The `observable:HolonomyGroup`
class captures the group of all holonomy transformations at a base point.

Monodromy — the representation of the fundamental group of the base space in the
holonomy group — is detected by the `observable:MonodromyResolver` and recorded in
`observable:MonodromyClass` individuals.

## Superposed Fiber States

A fiber need not be in a definite type state. The `type:SuperposedFiberState` and
`type:CollapsedFiberState` classes implement quantum superposition at the type level.

- A `SuperposedFiberState` is a linear combination of fiber types, weighted by amplitudes
- A `CollapsedFiberState` is the result of measurement — a definite type

This superposition is not merely formal. The `resolver:SuperpositionResolver` resolves
the superposed state, and the `cert:MeasurementCertificate` certifies that the collapse
event followed Born rule probabilities (documented in `cert:BornRuleVerification`).

## Partition Decomposition and the PRISM Pipeline

The partition structure mediates between the Kernel (Define stage) and the User types
(Resolve stage):

- **Kernel** defines the address space and ring arithmetic
- **Partition** (Bridge) decomposes the address space into fibers
- **Type** (User) assigns typed meaning to each fiber
- **Resolution** (Bridge) computes what type is at each address
- **Certification** (cert) certifies that the typing is consistent

The fiber bundle semantics is what makes UOR more than a list of algebraic facts — it is
a structured framework for typed computation over a content-addressed address space.
