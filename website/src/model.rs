//! Data model types for the website generator.

use serde::Serialize;

/// A single page in the website.
#[derive(Debug, Serialize)]
pub struct Page {
    /// Page title (without the " — UOR Foundation" suffix).
    pub title: String,
    /// Absolute path from site root (e.g. `/namespaces/schema/index.html`).
    pub path: String,
    /// HTML content of the page body.
    pub content: String,
    /// Breadcrumb trail.
    pub breadcrumbs: Vec<BreadcrumbItem>,
}

/// A breadcrumb navigation item.
#[derive(Debug, Serialize)]
pub struct BreadcrumbItem {
    /// Display label.
    pub label: String,
    /// URL (relative or absolute).
    pub url: String,
}

/// An entry in the JSON search index.
#[derive(Debug, Serialize)]
pub struct SearchEntry {
    /// Display label for search results.
    pub label: String,
    /// Short description / comment.
    pub description: String,
    /// URL to the page where this term is documented.
    pub url: String,
    /// Term kind: "class", "property", "individual", or "namespace".
    pub kind: String,
    /// Space classification: "kernel", "bridge", "user", or "".
    pub space: String,
    /// Namespace prefix (e.g. "op"), or "" for namespace entries themselves.
    pub namespace: String,
    /// Sub-kind: "enum" for enum classes, "identity" for identity individuals, else "".
    pub subkind: String,
}

/// Summary of a namespace for the homepage grid.
#[derive(Debug, Serialize)]
pub struct NamespaceSummary {
    /// Namespace prefix (e.g. `schema`).
    pub prefix: String,
    /// Namespace IRI.
    pub iri: String,
    /// Short label.
    pub label: String,
    /// Comment/description.
    pub comment: String,
    /// Space classification: "kernel", "user", or "bridge".
    pub space: String,
    /// URL to the namespace page.
    pub url: String,
    /// Class count.
    pub class_count: usize,
    /// Property count.
    pub property_count: usize,
    /// Individual count.
    pub individual_count: usize,
}

/// A navigation item (possibly with children).
#[derive(Debug, Serialize)]
pub struct NavItem {
    /// Display label.
    pub label: String,
    /// URL (empty string if this is a group heading).
    pub url: String,
    /// Child items.
    pub children: Vec<NavItem>,
}

/// Metadata for a concept deep-dive page.
#[derive(Debug, Serialize)]
pub struct ConceptPage {
    /// URL slug (e.g. `"ring"`).
    pub slug: String,
    /// Display title.
    pub title: String,
    /// One-sentence description shown in the concept grid.
    pub description: String,
    /// Full URL to the page.
    pub url: String,
    /// Space color coding: `"kernel"`, `"bridge"`, `"user"`, or `"cert"`.
    pub space: String,
}

/// An `op:Identity` individual for the identities browser.
#[derive(Debug, Serialize)]
pub struct IdentityEntry {
    /// Short local ID (e.g. `"D_1"`).
    pub id: String,
    /// Human-readable label.
    pub label: String,
    /// Description / comment.
    pub comment: String,
    /// Verification domain labels (may be multiple).
    pub domains: Vec<String>,
    /// URL anchor on the op namespace page.
    pub url: String,
}

/// A directed dependency edge between two namespaces (for the SVG graph).
#[derive(Debug, Serialize)]
pub struct NamespaceDependency {
    /// Importing namespace prefix.
    pub from_prefix: String,
    /// Imported namespace prefix.
    pub to_prefix: String,
}
