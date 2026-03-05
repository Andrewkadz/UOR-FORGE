//! `proof/` namespace — Verification proof structures.
//!
//! Proofs are kernel-produced attestations of algebraic properties. The
//! critical proof asserts the foundational theorem `neg(bnot(x)) = succ(x)`.
//!
//! **Space classification:** `bridge` — kernel-produced, user-consumed.

use crate::model::iris::*;
use crate::model::{
    Class, Individual, IndividualValue, Namespace, NamespaceModule, Property, PropertyKind, Space,
};

/// Returns the `proof/` namespace module.
#[must_use]
pub fn module() -> NamespaceModule {
    NamespaceModule {
        namespace: Namespace {
            prefix: "proof",
            iri: NS_PROOF,
            label: "UOR Proofs",
            comment: "Kernel-produced verification proofs attesting to algebraic \
                      properties of UOR objects and operations.",
            space: Space::Bridge,
            imports: &[NS_SCHEMA, NS_OP, NS_DERIVATION],
        },
        classes: classes(),
        properties: properties(),
        individuals: individuals(),
    }
}

fn classes() -> Vec<Class> {
    vec![
        Class {
            id: "https://uor.foundation/proof/Proof",
            label: "Proof",
            comment: "A kernel-produced attestation that a given algebraic property \
                      holds. The root class for all proof types.",
            subclass_of: &[OWL_THING],
            disjoint_with: &[],
        },
        Class {
            id: "https://uor.foundation/proof/CoherenceProof",
            label: "CoherenceProof",
            comment: "A proof of coherence: the type system and ring structure are \
                      mutually consistent at a given quantum level.",
            subclass_of: &["https://uor.foundation/proof/Proof"],
            disjoint_with: &[],
        },
        // v3.2: Proof modality classes
        Class {
            id: "https://uor.foundation/proof/ComputationCertificate",
            label: "ComputationCertificate",
            comment: "A proof confirmed by exhaustive execution over R_n at a specific \
                      quantum level. The kernel ran the identity against all 2^n inputs \
                      and observed that it holds. The proof:atQuantumLevel property \
                      records the level; proof:witness links to the WitnessData. \
                      CriticalIdentityProof is a subclass of ComputationCertificate.",
            subclass_of: &["https://uor.foundation/proof/Proof"],
            disjoint_with: &[],
        },
        Class {
            id: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "AxiomaticDerivation",
            comment: "A proof that follows from previously established axioms or \
                      definitions by equational, structural, or topological reasoning. \
                      The proof:derivationWitness property links to a \
                      derivation:Derivation individual recording the rewrite chain. \
                      All pipeline, constraint, observable, and topological identities \
                      are AxiomaticDerivations.",
            subclass_of: &["https://uor.foundation/proof/Proof"],
            disjoint_with: &[],
        },
        Class {
            id: "https://uor.foundation/proof/CriticalIdentityProof",
            label: "CriticalIdentityProof",
            comment: "A proof of the critical identity: neg(bnot(x)) = succ(x) \
                      for all x in R_n. This is the foundational theorem of the \
                      UOR kernel.",
            subclass_of: &["https://uor.foundation/proof/ComputationCertificate"],
            disjoint_with: &[],
        },
        Class {
            id: "https://uor.foundation/proof/WitnessData",
            label: "WitnessData",
            comment: "Supporting data for a proof: specific examples, counter-examples \
                      checked, or intermediate computation results.",
            subclass_of: &[OWL_THING],
            disjoint_with: &[],
        },
    ]
}

fn properties() -> Vec<Property> {
    vec![
        Property {
            id: "https://uor.foundation/proof/verified",
            label: "verified",
            comment: "Whether this proof has been verified by the kernel.",
            kind: PropertyKind::Datatype,
            functional: true,
            domain: Some("https://uor.foundation/proof/Proof"),
            range: XSD_BOOLEAN,
        },
        Property {
            id: "https://uor.foundation/proof/timestamp",
            label: "timestamp",
            comment: "The time at which this proof was produced.",
            kind: PropertyKind::Datatype,
            functional: true,
            domain: Some("https://uor.foundation/proof/Proof"),
            range: XSD_DATETIME,
        },
        Property {
            id: "https://uor.foundation/proof/witness",
            label: "witness",
            comment: "Supporting witness data for this proof.",
            kind: PropertyKind::Object,
            functional: false,
            domain: Some("https://uor.foundation/proof/Proof"),
            range: "https://uor.foundation/proof/WitnessData",
        },
        Property {
            id: "https://uor.foundation/proof/criticalIdentity",
            label: "criticalIdentity",
            comment: "Human-readable statement of the critical identity proven. \
                      E.g., 'neg(bnot(x)) = succ(x) for all x in R_n'. \
                      Annotation only — proof:provesIdentity is the typed reference.",
            kind: PropertyKind::Annotation,
            functional: true,
            domain: Some("https://uor.foundation/proof/CriticalIdentityProof"),
            range: XSD_STRING,
        },
        Property {
            id: "https://uor.foundation/proof/x",
            label: "x",
            comment: "A specific input value used as a witness for the critical \
                      identity check.",
            kind: PropertyKind::Datatype,
            functional: false,
            domain: Some("https://uor.foundation/proof/WitnessData"),
            range: XSD_INTEGER,
        },
        Property {
            id: "https://uor.foundation/proof/bnot_x",
            label: "bnot_x",
            comment: "The value bnot(x) for a witness x.",
            kind: PropertyKind::Datatype,
            functional: false,
            domain: Some("https://uor.foundation/proof/WitnessData"),
            range: XSD_INTEGER,
        },
        Property {
            id: "https://uor.foundation/proof/neg_bnot_x",
            label: "neg_bnot_x",
            comment: "The value neg(bnot(x)) for a witness x.",
            kind: PropertyKind::Datatype,
            functional: false,
            domain: Some("https://uor.foundation/proof/WitnessData"),
            range: XSD_INTEGER,
        },
        Property {
            id: "https://uor.foundation/proof/succ_x",
            label: "succ_x",
            comment: "The value succ(x) for a witness x.",
            kind: PropertyKind::Datatype,
            functional: false,
            domain: Some("https://uor.foundation/proof/WitnessData"),
            range: XSD_INTEGER,
        },
        Property {
            id: "https://uor.foundation/proof/holds",
            label: "holds",
            comment: "Whether the identity neg(bnot(x)) = succ(x) holds for \
                      this specific witness.",
            kind: PropertyKind::Datatype,
            functional: false,
            domain: Some("https://uor.foundation/proof/WitnessData"),
            range: XSD_BOOLEAN,
        },
        // Amendment 3: provesIdentity — object property linking to op:Identity
        Property {
            id: "https://uor.foundation/proof/provesIdentity",
            label: "provesIdentity",
            comment: "The algebraic identity this proof establishes. Provides a \
                      canonical object reference alongside the existing \
                      proof:criticalIdentity string property, which remains for \
                      human readability.",
            kind: PropertyKind::Object,
            functional: true,
            domain: Some("https://uor.foundation/proof/Proof"),
            range: "https://uor.foundation/op/Identity",
        },
        // v3.2: ComputationCertificate properties
        Property {
            id: "https://uor.foundation/proof/atQuantumLevel",
            label: "atQuantumLevel",
            comment: "The quantum level at which this computation certificate was \
                      produced. A ComputationCertificate at schema:Q0 confirms the \
                      identity holds for all 256 inputs of R_8. A certificate at \
                      schema:Q1 confirms it for all 65,536 inputs of R_16.",
            kind: PropertyKind::Object,
            functional: true,
            domain: Some("https://uor.foundation/proof/ComputationCertificate"),
            range: "https://uor.foundation/schema/QuantumLevel",
        },
        Property {
            id: "https://uor.foundation/proof/quantumNote",
            label: "quantumNote",
            comment: "Human-readable quantum level note, e.g. 'n=8, 256 inputs'. \
                      Annotation only — proof:atQuantumLevel is the typed assertion.",
            kind: PropertyKind::Annotation,
            functional: true,
            domain: Some("https://uor.foundation/proof/ComputationCertificate"),
            range: XSD_STRING,
        },
        // v3.2: AxiomaticDerivation properties
        Property {
            id: "https://uor.foundation/proof/universalScope",
            label: "universalScope",
            comment: "True when this axiomatic derivation holds for all quantum levels \
                      by the definition of Z/(2^n)Z. False when the derivation depends \
                      on a property specific to a particular ring size. All current \
                      AxiomaticDerivation individuals in the spec carry universalScope \
                      true.",
            kind: PropertyKind::Datatype,
            functional: true,
            domain: Some("https://uor.foundation/proof/AxiomaticDerivation"),
            range: XSD_BOOLEAN,
        },
        Property {
            id: "https://uor.foundation/proof/derivationWitness",
            label: "derivationWitness",
            comment: "The derivation chain that witnesses this axiomatic derivation. \
                      Links a proof:AxiomaticDerivation to the derivation:Derivation \
                      individual recording the rewrite sequence. Optional at the spec \
                      level — the conformance suite requires only that the proof \
                      individual exists; full derivation chains live in generated \
                      artifacts.",
            kind: PropertyKind::Object,
            functional: false,
            domain: Some("https://uor.foundation/proof/AxiomaticDerivation"),
            range: "https://uor.foundation/derivation/Derivation",
        },
    ]
}

fn individuals() -> Vec<Individual> {
    vec![
        // Critical identity: computation certificate (CriticalIdentityProof at Q0)
        Individual {
            id: "https://uor.foundation/proof/prf_criticalIdentity",
            type_: "https://uor.foundation/proof/CriticalIdentityProof",
            label: "prf_criticalIdentity",
            comment: "Computation certificate for the critical identity \
                      neg(bnot(x)) = succ(x) at Q0.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/criticalIdentity"),
                ),
                (
                    "https://uor.foundation/proof/atQuantumLevel",
                    IndividualValue::IriRef("https://uor.foundation/schema/Q0"),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        // Critical identity: axiomatic derivation (dual proof)
        Individual {
            id: "https://uor.foundation/proof/prf_criticalIdentity_axiomatic",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_criticalIdentity_axiomatic",
            comment: "Axiomatic derivation of the critical identity \
                      neg(bnot(x)) = succ(x). Holds at all quantum levels.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/criticalIdentity"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        // phi_1 through phi_6: computation certificates at Q0
        Individual {
            id: "https://uor.foundation/proof/prf_phi_1",
            type_: "https://uor.foundation/proof/ComputationCertificate",
            label: "prf_phi_1",
            comment: "Computation certificate for phi_1 at Q0.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/phi_1"),
                ),
                (
                    "https://uor.foundation/proof/atQuantumLevel",
                    IndividualValue::IriRef("https://uor.foundation/schema/Q0"),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_phi_2",
            type_: "https://uor.foundation/proof/ComputationCertificate",
            label: "prf_phi_2",
            comment: "Computation certificate for phi_2 at Q0.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/phi_2"),
                ),
                (
                    "https://uor.foundation/proof/atQuantumLevel",
                    IndividualValue::IriRef("https://uor.foundation/schema/Q0"),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_phi_3",
            type_: "https://uor.foundation/proof/ComputationCertificate",
            label: "prf_phi_3",
            comment: "Computation certificate for phi_3 at Q0.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/phi_3"),
                ),
                (
                    "https://uor.foundation/proof/atQuantumLevel",
                    IndividualValue::IriRef("https://uor.foundation/schema/Q0"),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_phi_4",
            type_: "https://uor.foundation/proof/ComputationCertificate",
            label: "prf_phi_4",
            comment: "Computation certificate for phi_4 at Q0.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/phi_4"),
                ),
                (
                    "https://uor.foundation/proof/atQuantumLevel",
                    IndividualValue::IriRef("https://uor.foundation/schema/Q0"),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_phi_5",
            type_: "https://uor.foundation/proof/ComputationCertificate",
            label: "prf_phi_5",
            comment: "Computation certificate for phi_5 at Q0.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/phi_5"),
                ),
                (
                    "https://uor.foundation/proof/atQuantumLevel",
                    IndividualValue::IriRef("https://uor.foundation/schema/Q0"),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_phi_6",
            type_: "https://uor.foundation/proof/ComputationCertificate",
            label: "prf_phi_6",
            comment: "Computation certificate for phi_6 at Q0.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/phi_6"),
                ),
                (
                    "https://uor.foundation/proof/atQuantumLevel",
                    IndividualValue::IriRef("https://uor.foundation/schema/Q0"),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        // Axiomatic derivations for all remaining op identities
        Individual {
            id: "https://uor.foundation/proof/prf_AD_1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_AD_1",
            comment: "Axiomatic derivation of AD_1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/AD_1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_AD_2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_AD_2",
            comment: "Axiomatic derivation of AD_2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/AD_2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_R_A1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_R_A1",
            comment: "Axiomatic derivation of R_A1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/R_A1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_R_A2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_R_A2",
            comment: "Axiomatic derivation of R_A2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/R_A2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_R_A3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_R_A3",
            comment: "Axiomatic derivation of R_A3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/R_A3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_R_A4",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_R_A4",
            comment: "Axiomatic derivation of R_A4. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/R_A4"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_R_A5",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_R_A5",
            comment: "Axiomatic derivation of R_A5. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/R_A5"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_R_A6",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_R_A6",
            comment: "Axiomatic derivation of R_A6. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/R_A6"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_R_M1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_R_M1",
            comment: "Axiomatic derivation of R_M1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/R_M1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_R_M2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_R_M2",
            comment: "Axiomatic derivation of R_M2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/R_M2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_R_M3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_R_M3",
            comment: "Axiomatic derivation of R_M3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/R_M3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_R_M4",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_R_M4",
            comment: "Axiomatic derivation of R_M4. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/R_M4"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_R_M5",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_R_M5",
            comment: "Axiomatic derivation of R_M5. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/R_M5"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_B_1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_B_1",
            comment: "Axiomatic derivation of B_1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/B_1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_B_2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_B_2",
            comment: "Axiomatic derivation of B_2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/B_2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_B_3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_B_3",
            comment: "Axiomatic derivation of B_3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/B_3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_B_4",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_B_4",
            comment: "Axiomatic derivation of B_4. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/B_4"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_B_5",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_B_5",
            comment: "Axiomatic derivation of B_5. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/B_5"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_B_6",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_B_6",
            comment: "Axiomatic derivation of B_6. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/B_6"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_B_7",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_B_7",
            comment: "Axiomatic derivation of B_7. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/B_7"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_B_8",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_B_8",
            comment: "Axiomatic derivation of B_8. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/B_8"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_B_9",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_B_9",
            comment: "Axiomatic derivation of B_9. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/B_9"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_B_10",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_B_10",
            comment: "Axiomatic derivation of B_10. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/B_10"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_B_11",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_B_11",
            comment: "Axiomatic derivation of B_11. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/B_11"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_B_12",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_B_12",
            comment: "Axiomatic derivation of B_12. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/B_12"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_B_13",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_B_13",
            comment: "Axiomatic derivation of B_13. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/B_13"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_X_1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_X_1",
            comment: "Axiomatic derivation of X_1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/X_1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_X_2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_X_2",
            comment: "Axiomatic derivation of X_2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/X_2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_X_3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_X_3",
            comment: "Axiomatic derivation of X_3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/X_3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_X_4",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_X_4",
            comment: "Axiomatic derivation of X_4. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/X_4"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_X_5",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_X_5",
            comment: "Axiomatic derivation of X_5. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/X_5"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_X_6",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_X_6",
            comment: "Axiomatic derivation of X_6. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/X_6"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_X_7",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_X_7",
            comment: "Axiomatic derivation of X_7. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/X_7"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_D_1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_D_1",
            comment: "Axiomatic derivation of D_1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/D_1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_D_3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_D_3",
            comment: "Axiomatic derivation of D_3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/D_3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_D_4",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_D_4",
            comment: "Axiomatic derivation of D_4. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/D_4"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_D_5",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_D_5",
            comment: "Axiomatic derivation of D_5. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/D_5"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_U_1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_U_1",
            comment: "Axiomatic derivation of U_1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/U_1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_U_2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_U_2",
            comment: "Axiomatic derivation of U_2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/U_2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_U_3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_U_3",
            comment: "Axiomatic derivation of U_3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/U_3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_U_4",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_U_4",
            comment: "Axiomatic derivation of U_4. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/U_4"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_U_5",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_U_5",
            comment: "Axiomatic derivation of U_5. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/U_5"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_AG_1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_AG_1",
            comment: "Axiomatic derivation of AG_1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/AG_1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_AG_2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_AG_2",
            comment: "Axiomatic derivation of AG_2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/AG_2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_AG_3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_AG_3",
            comment: "Axiomatic derivation of AG_3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/AG_3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_AG_4",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_AG_4",
            comment: "Axiomatic derivation of AG_4. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/AG_4"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_CA_1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_CA_1",
            comment: "Axiomatic derivation of CA_1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/CA_1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_CA_2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_CA_2",
            comment: "Axiomatic derivation of CA_2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/CA_2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_CA_3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_CA_3",
            comment: "Axiomatic derivation of CA_3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/CA_3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_CA_4",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_CA_4",
            comment: "Axiomatic derivation of CA_4. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/CA_4"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_CA_5",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_CA_5",
            comment: "Axiomatic derivation of CA_5. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/CA_5"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_CA_6",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_CA_6",
            comment: "Axiomatic derivation of CA_6. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/CA_6"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_C_1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_C_1",
            comment: "Axiomatic derivation of C_1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/C_1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_C_2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_C_2",
            comment: "Axiomatic derivation of C_2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/C_2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_C_3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_C_3",
            comment: "Axiomatic derivation of C_3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/C_3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_C_4",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_C_4",
            comment: "Axiomatic derivation of C_4. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/C_4"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_C_5",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_C_5",
            comment: "Axiomatic derivation of C_5. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/C_5"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_C_6",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_C_6",
            comment: "Axiomatic derivation of C_6. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/C_6"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_CDI",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_CDI",
            comment: "Axiomatic derivation of CDI. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/CDI"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_CL_1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_CL_1",
            comment: "Axiomatic derivation of CL_1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/CL_1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_CL_2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_CL_2",
            comment: "Axiomatic derivation of CL_2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/CL_2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_CL_3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_CL_3",
            comment: "Axiomatic derivation of CL_3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/CL_3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_CL_4",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_CL_4",
            comment: "Axiomatic derivation of CL_4. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/CL_4"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_CL_5",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_CL_5",
            comment: "Axiomatic derivation of CL_5. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/CL_5"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_CM_1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_CM_1",
            comment: "Axiomatic derivation of CM_1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/CM_1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_CM_2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_CM_2",
            comment: "Axiomatic derivation of CM_2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/CM_2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_CM_3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_CM_3",
            comment: "Axiomatic derivation of CM_3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/CM_3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_CR_1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_CR_1",
            comment: "Axiomatic derivation of CR_1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/CR_1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_CR_2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_CR_2",
            comment: "Axiomatic derivation of CR_2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/CR_2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_CR_3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_CR_3",
            comment: "Axiomatic derivation of CR_3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/CR_3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_CR_4",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_CR_4",
            comment: "Axiomatic derivation of CR_4. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/CR_4"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_CR_5",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_CR_5",
            comment: "Axiomatic derivation of CR_5. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/CR_5"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_F_1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_F_1",
            comment: "Axiomatic derivation of F_1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/F_1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_F_2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_F_2",
            comment: "Axiomatic derivation of F_2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/F_2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_F_3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_F_3",
            comment: "Axiomatic derivation of F_3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/F_3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_F_4",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_F_4",
            comment: "Axiomatic derivation of F_4. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/F_4"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_FL_1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_FL_1",
            comment: "Axiomatic derivation of FL_1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/FL_1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_FL_2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_FL_2",
            comment: "Axiomatic derivation of FL_2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/FL_2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_FL_3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_FL_3",
            comment: "Axiomatic derivation of FL_3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/FL_3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_FL_4",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_FL_4",
            comment: "Axiomatic derivation of FL_4. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/FL_4"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_FPM_1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_FPM_1",
            comment: "Axiomatic derivation of FPM_1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/FPM_1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_FPM_2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_FPM_2",
            comment: "Axiomatic derivation of FPM_2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/FPM_2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_FPM_3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_FPM_3",
            comment: "Axiomatic derivation of FPM_3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/FPM_3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_FPM_4",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_FPM_4",
            comment: "Axiomatic derivation of FPM_4. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/FPM_4"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_FPM_5",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_FPM_5",
            comment: "Axiomatic derivation of FPM_5. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/FPM_5"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_FPM_6",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_FPM_6",
            comment: "Axiomatic derivation of FPM_6. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/FPM_6"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_FPM_7",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_FPM_7",
            comment: "Axiomatic derivation of FPM_7. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/FPM_7"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_FS_1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_FS_1",
            comment: "Axiomatic derivation of FS_1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/FS_1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_FS_2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_FS_2",
            comment: "Axiomatic derivation of FS_2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/FS_2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_FS_3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_FS_3",
            comment: "Axiomatic derivation of FS_3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/FS_3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_FS_4",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_FS_4",
            comment: "Axiomatic derivation of FS_4. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/FS_4"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_FS_5",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_FS_5",
            comment: "Axiomatic derivation of FS_5. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/FS_5"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_FS_6",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_FS_6",
            comment: "Axiomatic derivation of FS_6. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/FS_6"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_FS_7",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_FS_7",
            comment: "Axiomatic derivation of FS_7. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/FS_7"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_RE_1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_RE_1",
            comment: "Axiomatic derivation of RE_1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/RE_1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_IR_1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_IR_1",
            comment: "Axiomatic derivation of IR_1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/IR_1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_IR_2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_IR_2",
            comment: "Axiomatic derivation of IR_2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/IR_2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_IR_3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_IR_3",
            comment: "Axiomatic derivation of IR_3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/IR_3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_IR_4",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_IR_4",
            comment: "Axiomatic derivation of IR_4. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/IR_4"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_SF_1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_SF_1",
            comment: "Axiomatic derivation of SF_1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/SF_1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_SF_2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_SF_2",
            comment: "Axiomatic derivation of SF_2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/SF_2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_RD_1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_RD_1",
            comment: "Axiomatic derivation of RD_1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/RD_1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_RD_2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_RD_2",
            comment: "Axiomatic derivation of RD_2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/RD_2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_SE_1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_SE_1",
            comment: "Axiomatic derivation of SE_1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/SE_1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_SE_2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_SE_2",
            comment: "Axiomatic derivation of SE_2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/SE_2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_SE_3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_SE_3",
            comment: "Axiomatic derivation of SE_3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/SE_3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_SE_4",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_SE_4",
            comment: "Axiomatic derivation of SE_4. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/SE_4"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_OO_1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_OO_1",
            comment: "Axiomatic derivation of OO_1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/OO_1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_OO_2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_OO_2",
            comment: "Axiomatic derivation of OO_2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/OO_2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_OO_3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_OO_3",
            comment: "Axiomatic derivation of OO_3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/OO_3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_OO_4",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_OO_4",
            comment: "Axiomatic derivation of OO_4. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/OO_4"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_OO_5",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_OO_5",
            comment: "Axiomatic derivation of OO_5. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/OO_5"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_CB_1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_CB_1",
            comment: "Axiomatic derivation of CB_1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/CB_1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_CB_2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_CB_2",
            comment: "Axiomatic derivation of CB_2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/CB_2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_CB_3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_CB_3",
            comment: "Axiomatic derivation of CB_3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/CB_3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_CB_4",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_CB_4",
            comment: "Axiomatic derivation of CB_4. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/CB_4"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_CB_5",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_CB_5",
            comment: "Axiomatic derivation of CB_5. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/CB_5"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_CB_6",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_CB_6",
            comment: "Axiomatic derivation of CB_6. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/CB_6"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_OB_M1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_OB_M1",
            comment: "Axiomatic derivation of OB_M1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/OB_M1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_OB_M2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_OB_M2",
            comment: "Axiomatic derivation of OB_M2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/OB_M2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_OB_M3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_OB_M3",
            comment: "Axiomatic derivation of OB_M3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/OB_M3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_OB_M4",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_OB_M4",
            comment: "Axiomatic derivation of OB_M4. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/OB_M4"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_OB_M5",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_OB_M5",
            comment: "Axiomatic derivation of OB_M5. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/OB_M5"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_OB_M6",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_OB_M6",
            comment: "Axiomatic derivation of OB_M6. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/OB_M6"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_OB_C1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_OB_C1",
            comment: "Axiomatic derivation of OB_C1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/OB_C1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_OB_C2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_OB_C2",
            comment: "Axiomatic derivation of OB_C2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/OB_C2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_OB_C3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_OB_C3",
            comment: "Axiomatic derivation of OB_C3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/OB_C3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_OB_H1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_OB_H1",
            comment: "Axiomatic derivation of OB_H1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/OB_H1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_OB_H2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_OB_H2",
            comment: "Axiomatic derivation of OB_H2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/OB_H2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_OB_H3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_OB_H3",
            comment: "Axiomatic derivation of OB_H3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/OB_H3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_OB_P1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_OB_P1",
            comment: "Axiomatic derivation of OB_P1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/OB_P1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_OB_P2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_OB_P2",
            comment: "Axiomatic derivation of OB_P2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/OB_P2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_OB_P3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_OB_P3",
            comment: "Axiomatic derivation of OB_P3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/OB_P3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_CT_1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_CT_1",
            comment: "Axiomatic derivation of CT_1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/CT_1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_CT_2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_CT_2",
            comment: "Axiomatic derivation of CT_2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/CT_2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_CT_3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_CT_3",
            comment: "Axiomatic derivation of CT_3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/CT_3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_CT_4",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_CT_4",
            comment: "Axiomatic derivation of CT_4. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/CT_4"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_CF_1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_CF_1",
            comment: "Axiomatic derivation of CF_1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/CF_1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_CF_2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_CF_2",
            comment: "Axiomatic derivation of CF_2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/CF_2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_CF_3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_CF_3",
            comment: "Axiomatic derivation of CF_3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/CF_3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_CF_4",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_CF_4",
            comment: "Axiomatic derivation of CF_4. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/CF_4"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_HG_1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_HG_1",
            comment: "Axiomatic derivation of HG_1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/HG_1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_HG_2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_HG_2",
            comment: "Axiomatic derivation of HG_2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/HG_2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_HG_3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_HG_3",
            comment: "Axiomatic derivation of HG_3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/HG_3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_HG_4",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_HG_4",
            comment: "Axiomatic derivation of HG_4. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/HG_4"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_HG_5",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_HG_5",
            comment: "Axiomatic derivation of HG_5. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/HG_5"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_T_C1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_T_C1",
            comment: "Axiomatic derivation of T_C1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/T_C1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_T_C2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_T_C2",
            comment: "Axiomatic derivation of T_C2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/T_C2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_T_C3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_T_C3",
            comment: "Axiomatic derivation of T_C3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/T_C3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_T_C4",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_T_C4",
            comment: "Axiomatic derivation of T_C4. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/T_C4"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_T_I1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_T_I1",
            comment: "Axiomatic derivation of T_I1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/T_I1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_T_I2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_T_I2",
            comment: "Axiomatic derivation of T_I2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/T_I2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_T_I3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_T_I3",
            comment: "Axiomatic derivation of T_I3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/T_I3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_T_I4",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_T_I4",
            comment: "Axiomatic derivation of T_I4. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/T_I4"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_T_I5",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_T_I5",
            comment: "Axiomatic derivation of T_I5. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/T_I5"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_T_E1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_T_E1",
            comment: "Axiomatic derivation of T_E1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/T_E1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_T_E2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_T_E2",
            comment: "Axiomatic derivation of T_E2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/T_E2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_T_E3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_T_E3",
            comment: "Axiomatic derivation of T_E3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/T_E3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_T_E4",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_T_E4",
            comment: "Axiomatic derivation of T_E4. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/T_E4"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_T_A1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_T_A1",
            comment: "Axiomatic derivation of T_A1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/T_A1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_T_A2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_T_A2",
            comment: "Axiomatic derivation of T_A2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/T_A2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_T_A3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_T_A3",
            comment: "Axiomatic derivation of T_A3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/T_A3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_T_A4",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_T_A4",
            comment: "Axiomatic derivation of T_A4. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/T_A4"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_AU_1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_AU_1",
            comment: "Axiomatic derivation of AU_1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/AU_1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_AU_2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_AU_2",
            comment: "Axiomatic derivation of AU_2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/AU_2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_AU_3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_AU_3",
            comment: "Axiomatic derivation of AU_3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/AU_3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_AU_4",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_AU_4",
            comment: "Axiomatic derivation of AU_4. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/AU_4"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_AU_5",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_AU_5",
            comment: "Axiomatic derivation of AU_5. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/AU_5"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_EF_1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_EF_1",
            comment: "Axiomatic derivation of EF_1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/EF_1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_EF_2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_EF_2",
            comment: "Axiomatic derivation of EF_2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/EF_2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_EF_3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_EF_3",
            comment: "Axiomatic derivation of EF_3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/EF_3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_EF_4",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_EF_4",
            comment: "Axiomatic derivation of EF_4. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/EF_4"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_EF_5",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_EF_5",
            comment: "Axiomatic derivation of EF_5. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/EF_5"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_EF_6",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_EF_6",
            comment: "Axiomatic derivation of EF_6. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/EF_6"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_EF_7",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_EF_7",
            comment: "Axiomatic derivation of EF_7. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/EF_7"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_AA_1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_AA_1",
            comment: "Axiomatic derivation of AA_1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/AA_1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_AA_2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_AA_2",
            comment: "Axiomatic derivation of AA_2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/AA_2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_AA_3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_AA_3",
            comment: "Axiomatic derivation of AA_3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/AA_3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_AA_4",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_AA_4",
            comment: "Axiomatic derivation of AA_4. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/AA_4"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_AA_5",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_AA_5",
            comment: "Axiomatic derivation of AA_5. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/AA_5"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_AA_6",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_AA_6",
            comment: "Axiomatic derivation of AA_6. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/AA_6"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_AM_1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_AM_1",
            comment: "Axiomatic derivation of AM_1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/AM_1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_AM_2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_AM_2",
            comment: "Axiomatic derivation of AM_2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/AM_2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_AM_3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_AM_3",
            comment: "Axiomatic derivation of AM_3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/AM_3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_AM_4",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_AM_4",
            comment: "Axiomatic derivation of AM_4. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/AM_4"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_TH_1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_TH_1",
            comment: "Axiomatic derivation of TH_1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/TH_1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_TH_2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_TH_2",
            comment: "Axiomatic derivation of TH_2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/TH_2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_TH_3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_TH_3",
            comment: "Axiomatic derivation of TH_3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/TH_3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_TH_4",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_TH_4",
            comment: "Axiomatic derivation of TH_4. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/TH_4"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_TH_5",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_TH_5",
            comment: "Axiomatic derivation of TH_5. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/TH_5"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_TH_6",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_TH_6",
            comment: "Axiomatic derivation of TH_6. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/TH_6"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_TH_7",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_TH_7",
            comment: "Axiomatic derivation of TH_7. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/TH_7"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_TH_8",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_TH_8",
            comment: "Axiomatic derivation of TH_8. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/TH_8"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_TH_9",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_TH_9",
            comment: "Axiomatic derivation of TH_9. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/TH_9"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_TH_10",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_TH_10",
            comment: "Axiomatic derivation of TH_10. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/TH_10"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_AR_1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_AR_1",
            comment: "Axiomatic derivation of AR_1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/AR_1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_AR_2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_AR_2",
            comment: "Axiomatic derivation of AR_2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/AR_2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_AR_3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_AR_3",
            comment: "Axiomatic derivation of AR_3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/AR_3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_AR_4",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_AR_4",
            comment: "Axiomatic derivation of AR_4. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/AR_4"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_AR_5",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_AR_5",
            comment: "Axiomatic derivation of AR_5. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/AR_5"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_PD_1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_PD_1",
            comment: "Axiomatic derivation of PD_1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/PD_1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_PD_2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_PD_2",
            comment: "Axiomatic derivation of PD_2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/PD_2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_PD_3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_PD_3",
            comment: "Axiomatic derivation of PD_3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/PD_3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_PD_4",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_PD_4",
            comment: "Axiomatic derivation of PD_4. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/PD_4"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_PD_5",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_PD_5",
            comment: "Axiomatic derivation of PD_5. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/PD_5"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_RC_1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_RC_1",
            comment: "Axiomatic derivation of RC_1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/RC_1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_RC_2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_RC_2",
            comment: "Axiomatic derivation of RC_2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/RC_2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_RC_3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_RC_3",
            comment: "Axiomatic derivation of RC_3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/RC_3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_RC_4",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_RC_4",
            comment: "Axiomatic derivation of RC_4. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/RC_4"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_RC_5",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_RC_5",
            comment: "Axiomatic derivation of RC_5. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/RC_5"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_DC_1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_DC_1",
            comment: "Axiomatic derivation of DC_1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/DC_1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_DC_2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_DC_2",
            comment: "Axiomatic derivation of DC_2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/DC_2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_DC_3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_DC_3",
            comment: "Axiomatic derivation of DC_3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/DC_3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_DC_4",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_DC_4",
            comment: "Axiomatic derivation of DC_4. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/DC_4"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_DC_5",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_DC_5",
            comment: "Axiomatic derivation of DC_5. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/DC_5"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_DC_6",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_DC_6",
            comment: "Axiomatic derivation of DC_6. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/DC_6"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_DC_7",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_DC_7",
            comment: "Axiomatic derivation of DC_7. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/DC_7"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_DC_8",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_DC_8",
            comment: "Axiomatic derivation of DC_8. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/DC_8"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_DC_9",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_DC_9",
            comment: "Axiomatic derivation of DC_9. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/DC_9"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_DC_10",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_DC_10",
            comment: "Axiomatic derivation of DC_10. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/DC_10"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_DC_11",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_DC_11",
            comment: "Axiomatic derivation of DC_11. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/DC_11"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_HA_1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_HA_1",
            comment: "Axiomatic derivation of HA_1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/HA_1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_HA_2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_HA_2",
            comment: "Axiomatic derivation of HA_2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/HA_2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_HA_3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_HA_3",
            comment: "Axiomatic derivation of HA_3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/HA_3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_IT_2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_IT_2",
            comment: "Axiomatic derivation of IT_2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/IT_2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_IT_3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_IT_3",
            comment: "Axiomatic derivation of IT_3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/IT_3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_IT_6",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_IT_6",
            comment: "Axiomatic derivation of IT_6. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/IT_6"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_IT_7a",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_IT_7a",
            comment: "Axiomatic derivation of IT_7a. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/IT_7a"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_IT_7b",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_IT_7b",
            comment: "Axiomatic derivation of IT_7b. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/IT_7b"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_IT_7c",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_IT_7c",
            comment: "Axiomatic derivation of IT_7c. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/IT_7c"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_IT_7d",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_IT_7d",
            comment: "Axiomatic derivation of IT_7d. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/IT_7d"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_psi_1",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_psi_1",
            comment: "Axiomatic derivation of psi_1. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/psi_1"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_psi_2",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_psi_2",
            comment: "Axiomatic derivation of psi_2. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/psi_2"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_psi_3",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_psi_3",
            comment: "Axiomatic derivation of psi_3. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/psi_3"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_psi_5",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_psi_5",
            comment: "Axiomatic derivation of psi_5. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/psi_5"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_psi_6",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_psi_6",
            comment: "Axiomatic derivation of psi_6. Holds at all quantum levels \
                      by definition of Z/(2^n)Z.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/op/psi_6"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        // Axiomatic derivations for homology identities
        Individual {
            id: "https://uor.foundation/proof/prf_boundarySquaredZero",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_boundarySquaredZero",
            comment: "Axiomatic derivation of homology:boundarySquaredZero. Holds at all quantum \
                      levels by topological reasoning.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/homology/boundarySquaredZero"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_psi_4",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_psi_4",
            comment: "Axiomatic derivation of homology:psi_4. Holds at all quantum \
                      levels by topological reasoning.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/homology/psi_4"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_indexBridge",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_indexBridge",
            comment: "Axiomatic derivation of homology:indexBridge. Holds at all quantum \
                      levels by topological reasoning.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/homology/indexBridge"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        // Axiomatic derivations for cohomology identities
        Individual {
            id: "https://uor.foundation/proof/prf_coboundarySquaredZero",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_coboundarySquaredZero",
            comment: "Axiomatic derivation of cohomology:coboundarySquaredZero. Holds at all \
                      quantum levels by topological reasoning.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef(
                        "https://uor.foundation/cohomology/coboundarySquaredZero",
                    ),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_deRhamDuality",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_deRhamDuality",
            comment: "Axiomatic derivation of cohomology:deRhamDuality. Holds at all \
                      quantum levels by topological reasoning.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef("https://uor.foundation/cohomology/deRhamDuality"),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_sheafCohomologyBridge",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_sheafCohomologyBridge",
            comment: "Axiomatic derivation of cohomology:sheafCohomologyBridge. Holds at all \
                      quantum levels by topological reasoning.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef(
                        "https://uor.foundation/cohomology/sheafCohomologyBridge",
                    ),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
        Individual {
            id: "https://uor.foundation/proof/prf_localGlobalPrinciple",
            type_: "https://uor.foundation/proof/AxiomaticDerivation",
            label: "prf_localGlobalPrinciple",
            comment: "Axiomatic derivation of cohomology:localGlobalPrinciple. Holds at all \
                      quantum levels by topological reasoning.",
            properties: &[
                (
                    "https://uor.foundation/proof/provesIdentity",
                    IndividualValue::IriRef(
                        "https://uor.foundation/cohomology/localGlobalPrinciple",
                    ),
                ),
                (
                    "https://uor.foundation/proof/universalScope",
                    IndividualValue::Bool(true),
                ),
                (
                    "https://uor.foundation/proof/verified",
                    IndividualValue::Bool(true),
                ),
            ],
        },
    ]
}
