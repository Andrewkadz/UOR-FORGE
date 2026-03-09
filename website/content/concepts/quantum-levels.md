# Quantum Levels

Quantum levels Q0–Q3 are the four scaling tiers of the UOR ring substrate. Every computation,
identity, and proof in UOR is valid at one or more quantum levels. Understanding quantum
levels is essential for reading the algebraic identities and their associated proofs.

## Level Definitions

| Level | Ring | Width | Cycle Size | Label |
|---|---|---|---|---|
| Q0 | Z/(2)Z | 1 bit | 2 | Boolean |
| Q1 | Z/(4)Z | 2 bits | 4 | Crumb |
| Q2 | Z/(16)Z | 4 bits | 16 | Nibble |
| Q3 | Z/(256)Z | 8 bits | 256 | Byte |

Each level is a `schema:QuantumLevel` named individual with:
- `schema:quantumIndex` — integer index 0–3
- `schema:bitsWidth` — bit width 1, 2, 4, or 8
- `schema:cycleSize` — ring cardinality (2^bitsWidth)
- `schema:nextLevel` — the next level up (Q0→Q1, Q1→Q2, Q2→Q3)

Q3 (the byte ring) is the primary ring for content addressing. Q0 is the minimal boolean
ring. The tower Q0 ⊂ Q1 ⊂ Q2 ⊂ Q3 is captured by the `nextLevel` chain.

## The `QuantumLevel` Enum

In the generated `uor-foundation` Rust crate, `QuantumLevel` is an open-world newtype
struct `QuantumLevel { index: u32 }` rather than a sealed enum. This is deliberate:
the OWL ontology treats quantum levels as a known finite set today, but the newtype
structure allows the framework to scale beyond Q3 without breaking changes.

The `#[non_exhaustive]` attribute on the Rust type signals this openness. The four
canonical instances Q0–Q3 are provided as associated constants.

## Universal vs. Level-Specific Validity

Not all algebraic identities hold at all quantum levels. The `op:ValidityScopeKind`
enum class captures four validity patterns:

- **Universal** — the identity holds for all quantum levels (k ∈ {0,1,2,3} and beyond)
- **ParametricLower** — the identity holds for all k ≥ k_min (some minimum level)
- **ParametricRange** — the identity holds for k_min ≤ k ≤ k_max
- **LevelSpecific** — the identity holds only at exactly one quantum level

These scopes are enforced by the conformance validator `validate_forall_scope_alignment`,
which checks that the `universallyValid` boolean is consistent with the `validityKind`
and that `validKMin`/`validKMax` bounds are present when required.

## The `ValidityScopeKind` Taxonomy

The four individuals of `ValidityScopeKind` are:

- `op:Universal` — represents the `∀k` quantifier
- `op:ParametricLower` — represents `∀k ≥ k_min`; requires `validKMin` property
- `op:ParametricRange` — represents `∀k ∈ [k_min, k_max]`; requires both bounds
- `op:LevelSpecific` — represents a claim about exactly one k

Most of the 378 named identities are Universal (they hold over Z/(2^n)Z for all n in the
tower). The level-specific identities capture phenomena that only emerge at particular
ring scales — for example, geometric properties that depend on having at least 4 elements
(requiring Q2 or above).

## Tower Chains

Amendments 41 and beyond introduced tower chain vocabulary to formalize the Q-n scaling
of inductive proofs. A `type:LiftChain` records a sequence of type lifts from level k to
level k+1, and a `type:LiftObstruction` records where the lifting fails.

The `cert:LiftChainCertificate` certifies that a lift chain is valid all the way to the
target level. The `type:ObstructionChain` captures the failure mode — where and why a
lift cannot proceed.

Inductive proofs (`proof:InductiveProof`) have three key properties:
- `proof:baseCase` — proof at the minimal quantum level
- `proof:inductiveStep` — proof that validity at level k implies validity at level k+1
- `proof:validForKAtLeast` — the certified minimum quantum level

This allows the tower chain structure to be formally certified in `cert`, closing the
inductive proof pathway through the PRISM pipeline.

## Connection to Content Addressing

The quantum level determines the granularity of content addressing. At Q3 (byte width),
a content address is a sequence of bytes — the standard interpretation for file systems,
network protocols, and cryptographic hash functions.

At Q0 (bit width), a content address is a single bit — useful for Boolean domains.
The ring Z/(2)Z has only two elements (0 and 1), making Q0 addresses the coarsest
possible content-addressable identifiers.

The `schema:QuantumLevel` individuals are referenced by the `op:QuantumLevelBinding` class,
which links operations to the quantum level at which they are defined. This binding ensures
that no operation is applied at a ring scale it was not designed for.
