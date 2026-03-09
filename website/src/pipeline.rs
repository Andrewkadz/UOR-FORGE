//! PRISM pipeline stage definitions and concept SVG hook registry.
//!
//! `PRISM_STAGES` is the single source of truth for pipeline stage names,
//! section IDs, and space/prefix matchers. All renderers and conformance
//! validators import from here — no scattered stage name literals.

use uor_ontology::Ontology;

/// A PRISM pipeline stage definition.
///
/// Fields: (display_name, section_id, match_key, is_prefix_match).
///
/// - When `is_prefix_match == false`, `match_key` is compared against a namespace's
///   `space` string (e.g., `"kernel"`, `"bridge"`).
/// - When `is_prefix_match == true`, `match_key` is compared against a namespace prefix
///   (e.g., `"cert"` — the cert namespace is Bridge space but belongs to the Certify stage).
pub const PRISM_STAGES: &[(&str, &str, &str, bool)] = &[
    ("Define", "stage-define", "kernel", false),
    ("Resolve", "stage-resolve", "bridge", false),
    ("Certify", "stage-certify", "cert", true),
];

/// SVG hook function type: takes an ontology reference, returns an SVG string.
pub type SvgHookFn = fn(&Ontology) -> String;

/// Maps concept page slugs to SVG generator functions.
///
/// When generating a concept page, if a `(slug, fn)` entry exists for the concept's
/// slug, the generator is called and its SVG is injected after the page `<h1>`.
/// This eliminates `match` expressions on hard-coded slug strings in `generate()`.
pub const CONCEPT_SVG_HOOKS: &[(&str, SvgHookFn)] = &[
    ("quantum-levels", crate::svg::render_quantum_levels_svg),
    ("prism", crate::svg::render_prism_pipeline_svg_for_concept),
];
