//! UOR Foundation ontology encoded as typed Rust data.
//!
//! The `uor-ontology` crate provides the complete UOR Foundation ontology —
//! 16 namespaces, 155 classes, 285 properties, and 597 named individuals —
//! as static Rust data structures, along with serializers that produce
//! JSON-LD, Turtle, and N-Triples output.
//!
//! # Entry Point
//!
//! ```
//! let ontology = uor_ontology::Ontology::full();
//! assert_eq!(ontology.namespaces.len(), 16);
//! ```
//!
//! # Serialization
//!
//! Requires the `serializers` feature (enabled by default).
//!
//! ```
//! let ontology = uor_ontology::Ontology::full();
//! let json_ld = uor_ontology::serializer::jsonld::to_json_ld(ontology);
//! let turtle  = uor_ontology::serializer::turtle::to_turtle(ontology);
//! ```
//!
//! # Feature Flags
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `serde` | yes | Adds `Serialize` derive to all model types |
//! | `serializers` | yes | JSON-LD, Turtle, and N-Triples serializers (pulls in `serde_json`) |
//!
//! This crate is internal (not published). The published crate `uor-foundation`
//! is generated from this data by the `uor-crate` client.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    missing_docs,
    clippy::missing_errors_doc
)]

pub mod model;
pub mod namespaces;
#[cfg(feature = "serializers")]
pub mod serializer;

pub use model::iris;
pub use model::{
    AnnotationProperty, Class, Individual, IndividualValue, Namespace, NamespaceModule, Ontology,
    Property, PropertyKind, Space,
};

impl Ontology {
    /// Returns the complete UOR Foundation ontology with all 16 namespaces
    /// and all 27 amendments applied.
    ///
    /// Assembly order follows the dependency graph specified in the UOR Foundation
    /// completion plan:
    /// `u → schema → op → query → resolver → type → partition →
    ///  observable → homology → cohomology → proof → derivation → trace → cert → morphism → state`
    #[must_use]
    pub fn full() -> &'static Ontology {
        static ONTOLOGY: std::sync::OnceLock<Ontology> = std::sync::OnceLock::new();
        ONTOLOGY.get_or_init(|| Ontology {
            version: "3.4.0",
            base_iri: "https://uor.foundation/",
            namespaces: vec![
                namespaces::u::module(),
                namespaces::schema::module(),
                namespaces::op::module(),
                namespaces::query::module(),
                namespaces::resolver::module(),
                namespaces::type_::module(),
                namespaces::partition::module(),
                namespaces::observable::module(),
                namespaces::homology::module(),
                namespaces::cohomology::module(),
                namespaces::proof::module(),
                namespaces::derivation::module(),
                namespaces::trace::module(),
                namespaces::cert::module(),
                namespaces::morphism::module(),
                namespaces::state::module(),
            ],
            annotation_properties: vec![model::annotation_space_property()],
        })
    }

    /// Looks up a namespace module by its short prefix (e.g., `"u"`, `"schema"`).
    #[must_use]
    pub fn find_namespace(&self, prefix: &str) -> Option<&NamespaceModule> {
        self.namespaces
            .iter()
            .find(|m| m.namespace.prefix == prefix)
    }

    /// Looks up a namespace module by its full IRI (e.g., `"https://uor.foundation/u/"`).
    #[must_use]
    pub fn find_namespace_by_iri(&self, iri: &str) -> Option<&NamespaceModule> {
        self.namespaces.iter().find(|m| m.namespace.iri == iri)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn namespace_count() {
        assert_eq!(Ontology::full().namespaces.len(), 16);
    }

    #[test]
    fn class_count() {
        let total: usize = Ontology::full()
            .namespaces
            .iter()
            .map(|m| m.classes.len())
            .sum();
        // 155 classes: 142 v3.3.0 + 13 Amdt25-27 (CompletenessCandidate, CompletenessWitness,
        // CompletenessAuditTrail, CompletenessResolver, QuantumLevelResolver, SessionResolver,
        // Q1Ring, QuantumLevelBinding, SessionBoundaryType, Session, BindingAccumulator,
        // SessionBoundary, SessionQuery).
        assert_eq!(total, 155);
    }

    #[test]
    fn property_count() {
        // 285 = 262 v3.3.0 + 23 Amdt25-27 (4 type + 2 cert + 3 resolver + 2 schema + 3 op + 8 state + 1 query).
        assert_eq!(Ontology::full().property_count(), 285);
    }

    #[test]
    fn individual_count() {
        let total: usize = Ontology::full()
            .namespaces
            .iter()
            .map(|m| m.individuals.len())
            .sum();
        // 597 individuals: 560 v3.3.0 + 20 Amdt25-27 (17 op identities + 3 state SessionBoundaryType)
        //                + 17 proof individuals for CC_1–5, QL_1–7, SR_1–5.
        assert_eq!(total, 597);
    }

    #[test]
    fn all_class_iris_unique() {
        let mut iris = std::collections::HashSet::new();
        for module in &Ontology::full().namespaces {
            for class in &module.classes {
                assert!(iris.insert(class.id), "Duplicate class IRI: {}", class.id);
            }
        }
    }

    #[test]
    fn all_property_iris_unique() {
        let mut iris = std::collections::HashSet::new();
        for module in &Ontology::full().namespaces {
            for prop in &module.properties {
                assert!(iris.insert(prop.id), "Duplicate property IRI: {}", prop.id);
            }
        }
    }

    #[test]
    fn all_individual_iris_unique() {
        let mut iris = std::collections::HashSet::new();
        for module in &Ontology::full().namespaces {
            for ind in &module.individuals {
                assert!(iris.insert(ind.id), "Duplicate individual IRI: {}", ind.id);
            }
        }
    }

    #[test]
    fn space_annotations_on_all_namespaces() {
        for module in &Ontology::full().namespaces {
            // Every namespace must have a space classification.
            let _ = &module.namespace.space; // Space is non-optional; this compiles only if present.
        }
    }

    #[test]
    fn find_namespace_by_prefix() {
        let ontology = Ontology::full();
        let u = ontology.find_namespace("u");
        assert!(u.is_some());
        assert_eq!(
            u.map(|m| m.namespace.iri),
            Some("https://uor.foundation/u/")
        );
    }

    #[test]
    fn find_namespace_by_iri_works() {
        let ontology = Ontology::full();
        let schema = ontology.find_namespace_by_iri("https://uor.foundation/schema/");
        assert!(schema.is_some());
        assert_eq!(schema.map(|m| m.namespace.prefix), Some("schema"));
    }
}
