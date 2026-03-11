# FORGE.md — UOR-FORGE Master Project Specification

> **UOR-FORGE** is a semantically-addressed multi-teacher model distillation engine. It extracts the highest-signal reasoning vectors from multiple frontier 120B language models and fuses them into a single deployable 7B student model — **AMALGEMEN** — whose every capability is traceable to a semantic coordinate in the UOR topology.

---

## 1. Purpose and Motivation

Standard model merging is mathematically blind. Existing techniques — SLERP, TIES, DARE, and their variants — operate on raw weight magnitudes with no understanding of what those weights encode. They treat tensors as numerical objects, not as semantic carriers. The result is a merged model whose capabilities are statistically averaged but semantically unaddressed: you cannot ask *what* was merged, *from where*, or *why*.

UOR-FORGE introduces a fundamentally different approach. Every extracted reasoning vector carries a **meaning address** — a coordinate in the Universal Object Reference (UOR) topology — before any merge decision is made. Fusion is guided by semantic proximity in that coordinate space, not by weight magnitude alone. The output is not just a smaller model; it is a model with a fully documented semantic map of what it knows and where each capability originated.

This is the first model distillation system of its kind to be semantically addressed rather than mathematically assembled.

---

## 2. System Overview

UOR-FORGE is composed of three interlocking layers, each with a distinct role and a strict boundary:

| Layer | Component | Role |
|---|---|---|
| **Coordinate Substrate** | `foundation/` — cat9 topology | Frozen UOR coordinate system; semantic ground truth |
| **Operator Layer** | HARMONIA-DSL | Language for expressing and manipulating semantic relationships between vectors |
| **Output** | AMALGEMEN (7B) | Distilled student model with a fully documented semantic map |

These layers are not interchangeable. The substrate is immutable. The operator layer is the only sanctioned interface for expressing merge logic. The output is the only artifact that leaves the forge.

---

## 3. The Coordinate Substrate — cat9 Topology

The `/foundation/` directory contains the **cat9 topology**: the frozen UOR coordinate system that all operations in this repository are anchored to. It is never modified during distillation. It is the semantic ground truth of the project.

The topology is formally grounded in the UOR Foundation ontology (v6.3.0), which encodes:

| Dimension | Count |
|---|---|
| Namespaces | 16 |
| OWL Classes | 218 |
| OWL Properties | 446 |
| Named Individuals | 846 |
| Algebraic Identities | 378 |
| Amendments Applied | 49 |

### 3.1 Ring Substrate

The algebraic foundation is the ring **Z/(2^n)Z**, equipped with two involutions:

- **Ring negation** — `neg(x) = (-x) mod 2^n`
- **Bitwise complement** — `bnot(x) = (2^n - 1) ⊕ x`

Their interaction is governed by the critical identity:

```
neg(bnot(x)) = succ(x)
```

This identity generates the **dihedral group D_{2^n}**, which is the algebraic engine underlying all resolution and factorization operations in the topology.

### 3.2 Namespace Tripartition

All 16 namespaces are classified into three spaces, reflecting their role in the resolution pipeline:

| Space | Namespaces | Role |
|---|---|---|
| **Kernel** | 3 (`u/`, `schema/`, `op/`) | Immutable; compiled into ROM; addressing, schema, primitive operations |
| **Bridge** | 10 | Kernel-computed, user-consumed; queries, resolution, partitions, proofs, traces, certificates |
| **User** | 3 (`type/`, `morphism/`, `state/`) | Runtime declarations; types, transforms, context |

### 3.3 Resolution Pipeline

The UOR resolution pipeline is the formal process by which a typed value is transformed into a certified, traceable result. In the context of UOR-FORGE, this pipeline is repurposed to resolve **semantic coordinates for reasoning vectors** extracted from teacher models.

The pipeline stages are:

1. **Define** — Declare a type with constraints (residue classes, carry patterns, depth bounds) that pin fibers of the iterated Z/2Z fibration.
2. **Resolve** — The resolver factorizes the element under the dihedral group, classifies it into the four-component partition (irreducible, reducible, unit, exterior), and measures observables.
3. **Certify** — A certificate attests the resolution result with a verification hash and computation trace, enabling independent replay.

### 3.4 Structural Reasoning

When constraint interactions produce topological obstructions, the pipeline employs algebraic topology to diagnose and resolve them:

- The **constraint nerve** (a simplicial complex) detects cyclic dependencies via Betti numbers.
- **Sheaf cohomology** (H^1) detects gluing failures — cases where local constraint satisfaction cannot be assembled into a global resolution.

These mechanisms apply equally to ontology resolution and to semantic vector fusion in AMALGEMEN distillation.

---

## 4. The Operator Layer — HARMONIA-DSL

HARMONIA-DSL is the semantic operator language that sits above the cat9 topology. It is the only sanctioned interface through which semantic relationships between reasoning vectors are expressed and manipulated.

HARMONIA-DSL operators are not weight-space operations. They are **coordinate-space operations** — they act on meaning addresses, not tensor indices. The DSL provides:

- **Semantic proximity operators** — measure distance between reasoning vectors in UOR coordinate space.
- **Fusion operators** — combine vectors from multiple teachers according to proximity and confidence scores.
- **Provenance operators** — tag every fused vector with its origin teacher and coordinate address.
- **Validation operators** — verify that a proposed fusion does not violate topological constraints in the cat9 substrate.

The grammar and operator set of HARMONIA-DSL are maintained in the [HARMONIA-DSL repository](https://github.com/Andrewkadz/HARMONIA-DSL). Changes to the DSL must not alter the cat9 topology.

---

## 5. The Distillation Engine

### 5.1 What Distillation Means Here

In UOR-FORGE, distillation is not knowledge distillation in the classical sense (soft-label transfer from teacher logits to student logits). It is **semantic vector extraction and fusion**: the process of identifying, addressing, and combining the highest-signal reasoning capabilities from multiple frontier teacher models into a single student model.

The key distinction is that every extracted capability is assigned a UOR coordinate before fusion. This means the student model does not merely inherit statistical patterns — it inherits **addressed capabilities** that can be audited, traced, and selectively updated.

### 5.2 Teacher Models

UOR-FORGE is designed to operate with multiple frontier 120B-class teacher models simultaneously. The distillation engine treats each teacher as a source of reasoning vectors, not as a monolithic artifact to be averaged. Teachers are selected for their complementary reasoning profiles — the goal is coverage of the semantic coordinate space, not redundancy.

### 5.3 The Fusion Process

The fusion process proceeds in four stages:

| Stage | Operation | UOR Mechanism |
|---|---|---|
| **Extraction** | Identify high-signal reasoning vectors from each teacher | Resolver pipeline maps vectors to coordinate addresses |
| **Addressing** | Assign UOR coordinates to each extracted vector | cat9 topology provides the address space |
| **Proximity Scoring** | Compute semantic distance between vectors from different teachers | HARMONIA-DSL proximity operators |
| **Fusion** | Combine vectors guided by proximity and confidence | HARMONIA-DSL fusion operators; certificate issued per fusion |

Every fusion decision is recorded in a **provenance trace** — a computation trace in the UOR sense — that links the fused vector in AMALGEMEN to its source teachers, its coordinate address, and the certificate that attests the fusion was valid.

### 5.4 What Makes This Different

| Property | Standard Merging | UOR-FORGE |
|---|---|---|
| Operates on | Weight magnitudes | Semantic coordinates |
| Merge guidance | Magnitude heuristics | Semantic proximity in UOR space |
| Auditability | None | Full provenance trace per vector |
| Topological safety | Not checked | Verified via constraint nerve and sheaf cohomology |
| Output documentation | None | Semantic map of all capabilities and their origins |

---

## 6. AMALGEMEN — The Student Model

**AMALGEMEN** is the 7B student model produced by UOR-FORGE. It is the primary output artifact of the forge.

AMALGEMEN is distinguished from other distilled or merged models by two properties:

1. **Semantic addressing** — Every capability in AMALGEMEN is traceable to a UOR coordinate in the cat9 topology. The model ships with a semantic map that documents what it knows and where each capability originated.

2. **Provenance completeness** — Every fused reasoning vector carries a certificate linking it to its source teacher(s), its coordinate address, and the HARMONIA-DSL operators used in fusion. This is not metadata appended after the fact; it is a first-class output of the distillation pipeline.

AMALGEMEN is the first model of its kind to be semantically addressed rather than mathematically assembled.

### 6.1 Target Specification

| Property | Value |
|---|---|
| Parameter count | 7B |
| Architecture | To be determined per teacher compatibility |
| Semantic map | Included as a structured artifact alongside model weights |
| Provenance format | UOR certificate chain (JSON-LD / Turtle serializable) |
| Coordinate substrate | cat9 topology v6.3.0 |

---

## 7. Repository Structure

```
UOR-FORGE/
├── foundation/          # cat9 topology — frozen UOR coordinate substrate
│   └── src/             # Generated Rust traits (uor-foundation crate)
├── spec/                # Ontology source of truth (uor-ontology crate)
│   └── src/
│       ├── namespaces/  # 16 namespace definitions
│       ├── model.rs     # Core ontology model types
│       └── counts.rs    # Authoritative inventory counts
├── codegen/             # Code generation logic for the Rust trait crate
├── conformance/         # Workspace-wide conformance validators
├── docs/                # Documentation generator
├── website/             # Static site generator
├── clients/             # CLI binaries: build, conformance, docs, website, crate
├── FORGE.md             # This document — master project specification
├── README.md            # Machine-generated ontology documentation
└── CLAUDE.md            # Developer workflow reference
```

The HARMONIA-DSL operator layer is maintained in a separate repository and linked to this forge via coordinate references, not file inclusion.

---

## 8. Invariants

The following invariants must hold at all times. Violations are build failures.

1. **The cat9 topology is frozen.** The `/foundation/` directory and the 16 namespaces in `/spec/src/namespaces/` are not modified during distillation operations. Any change to the topology requires a versioned amendment following the established amendment process (currently at Amendment 49).

2. **Every fused vector has a coordinate address.** No vector may enter AMALGEMEN without a valid UOR coordinate assigned by the cat9 topology. Unaddressed vectors are rejected at the fusion stage.

3. **Every fusion has a certificate.** No fusion decision is final without a UOR certificate attesting its validity. Certificates are not optional metadata; they are required outputs of the fusion operator.

4. **HARMONIA-DSL is the only fusion interface.** Direct weight-space operations that bypass the DSL are prohibited. All merge logic must be expressible as HARMONIA-DSL programs.

5. **Provenance is complete.** The semantic map shipped with AMALGEMEN must account for every capability in the model. Gaps in provenance are treated as build failures, not warnings.

---

## 9. Conformance and Validation

UOR-FORGE inherits the conformance infrastructure of the UOR Framework. The full conformance suite (`uor-conformance`) validates:

- Ontology structure (OWL 2 DL, SHACL shapes, JSON-LD, Turtle, N-Triples)
- Generated Rust trait crate (`#![no_std]` build, trait/method/enum counts)
- Documentation and website completeness

In addition, FORGE-specific conformance checks validate:

- Coordinate address completeness for all extracted vectors
- Certificate chain integrity for all fusion decisions
- Semantic map coverage against the full AMALGEMEN parameter space
- HARMONIA-DSL program validity against the cat9 topology

---

## 10. Roadmap

The following phases define the build sequence for UOR-FORGE:

| Phase | Milestone | Description |
|---|---|---|
| **P0** | Substrate lock | Confirm cat9 topology version; freeze `/foundation/` for distillation |
| **P1** | DSL integration | Connect HARMONIA-DSL operator layer to the forge pipeline |
| **P2** | Teacher ingestion | Define extraction protocol for 120B teacher models |
| **P3** | Addressing | Implement resolver pipeline for reasoning vector coordinate assignment |
| **P4** | Fusion | Implement HARMONIA-DSL fusion operators and certificate issuance |
| **P5** | Student training | Train AMALGEMEN 7B on fused, addressed vectors |
| **P6** | Semantic map | Generate and validate the full provenance semantic map |
| **P7** | Release | Ship AMALGEMEN weights + semantic map + certificate chain |

---

## 11. Definitions

| Term | Definition |
|---|---|
| **UOR** | Universal Object Reference — the formal mathematical framework for content-addressed, algebraically-structured object spaces |
| **cat9 topology** | The frozen UOR coordinate system in `/foundation/`; the semantic ground truth of the project |
| **HARMONIA-DSL** | The semantic operator language for expressing and manipulating relationships between reasoning vectors |
| **AMALGEMEN** | The 7B student model produced by UOR-FORGE; the primary output artifact |
| **Reasoning vector** | A high-signal capability extracted from a teacher model and assigned a UOR coordinate |
| **Semantic map** | The structured artifact documenting every capability in AMALGEMEN, its coordinate address, and its provenance |
| **Provenance trace** | A UOR computation trace linking a fused vector to its source teachers, coordinate, and certificate |
| **Fusion** | The HARMONIA-DSL-mediated process of combining reasoning vectors from multiple teachers into a single addressed vector |
| **Certificate** | A UOR attestation that a fusion decision is valid and reproducible |
| **Amendment** | A versioned change to the UOR ontology; the topology is currently at Amendment 49 |

---

*This document is the master specification for UOR-FORGE. It is the authoritative reference for all design decisions, invariants, and roadmap items. All contributors and automated systems operating within this repository are bound by the invariants stated in Section 8.*
