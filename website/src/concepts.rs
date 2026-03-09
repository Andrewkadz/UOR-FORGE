//! Concept page discovery and rendering.
//!
//! Concept pages are discovered dynamically from `content/concepts/*.md` files.
//! Adding a new `.md` file automatically produces a new HTML page — no hard-coded
//! slug list is maintained here.

use std::path::Path;

use anyhow::{Context, Result};

use crate::model::ConceptPage;

/// Discovers and returns all concept pages from the `content/concepts/` directory.
///
/// Pages are sorted alphabetically by slug for deterministic output order.
///
/// # Errors
///
/// Returns an error if the concepts directory cannot be read.
pub fn concept_page_list(content_dir: &Path, base_path: &str) -> Result<Vec<ConceptPage>> {
    let concepts_dir = content_dir.join("concepts");
    if !concepts_dir.exists() {
        return Ok(Vec::new());
    }

    let mut entries: Vec<std::fs::DirEntry> = std::fs::read_dir(&concepts_dir)
        .with_context(|| format!("Failed to read {}", concepts_dir.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "md").unwrap_or(false))
        .collect();

    entries.sort_by_key(|e| e.file_name());

    entries
        .iter()
        .map(|entry| {
            let slug = entry
                .path()
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();
            let raw = std::fs::read_to_string(entry.path())
                .with_context(|| format!("Failed to read {}", entry.path().display()))?;
            let title = first_h1(&raw).unwrap_or_else(|| slug.replace('-', " "));
            let description = first_paragraph(&raw).unwrap_or_default();
            let space = infer_space(&slug);
            Ok(ConceptPage {
                url: format!("{base_path}/concepts/{slug}.html"),
                slug,
                title,
                description,
                space,
            })
        })
        .collect()
}

/// Reads and renders a concept markdown file to an HTML body string.
///
/// Uses `pulldown_cmark` for CommonMark rendering.
///
/// # Errors
///
/// Returns an error if the content file cannot be read.
pub fn render_concept_from_file(path: &Path) -> Result<String> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read concept file {}", path.display()))?;
    Ok(markdown_to_html(&raw))
}

/// Converts CommonMark markdown to HTML.
fn markdown_to_html(source: &str) -> String {
    use pulldown_cmark::{html, Options, Parser};
    let opts = Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TABLES;
    let parser = Parser::new_ext(source, opts);
    let mut html_output = String::new();
    html::push_html(&mut html_output, parser);
    html_output
}

/// Extracts the first ATX heading (`# Title`) from raw markdown.
fn first_h1(md: &str) -> Option<String> {
    md.lines()
        .find(|l| l.starts_with("# "))
        .map(|l| l.trim_start_matches("# ").trim().to_string())
}

/// Extracts the first non-heading, non-empty paragraph from raw markdown.
fn first_paragraph(md: &str) -> Option<String> {
    md.lines()
        .skip_while(|l| l.starts_with('#') || l.trim().is_empty())
        .take_while(|l| !l.trim().is_empty())
        .collect::<Vec<_>>()
        .join(" ")
        .into()
}

/// Infers the space classification for a concept page from its slug.
///
/// This is editorial metadata for color-coding cards — not a strict ontology rule.
fn infer_space(slug: &str) -> String {
    match slug {
        "ring" | "quantum-levels" | "prism" => "kernel".to_string(),
        "fiber" => "bridge".to_string(),
        _ => "kernel".to_string(),
    }
}
