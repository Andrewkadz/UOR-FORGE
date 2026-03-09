//! UOR Foundation static site generator.
//!
//! Generates the complete https://uor.foundation/ website as a directory of
//! static HTML, CSS, and JavaScript files. All namespace and class pages are
//! 100% auto-generated from `uor_ontology::Ontology::full()`.
//!
//! # Entry Point
//!
//! ```no_run
//! use std::path::PathBuf;
//! use uor_website::generate;
//!
//! let out = PathBuf::from("public");
//! generate(&out).expect("Website generation failed");
//! ```
//!
//! # Output Structure
//!
//! ```text
//! public/
//!   index.html
//!   search.html
//!   search-index.json
//!   sitemap.xml
//!   pipeline/index.html
//!   explore/index.html
//!   identities/index.html
//!   download/index.html
//!   concepts/index.html
//!   concepts/<slug>.html   (one per content/concepts/*.md file)
//!   namespaces/
//!     <prefix>/index.html  (one per namespace, 100% auto-generated)
//!   css/style.css
//!   js/search.js
//! ```
//!
//! # Base Path
//!
//! Set the `PUBLIC_BASE_PATH` environment variable to a path prefix (e.g.
//! `/UOR-Framework`) when the site is served from a subdirectory. Defaults to
//! an empty string (served from the root).

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    missing_docs,
    clippy::missing_errors_doc
)]

pub mod concepts;
pub mod extractor;
pub mod model;
pub mod nav;
pub mod pipeline;
pub mod renderer;
pub mod search;
pub mod svg;
pub mod writer;

use std::path::Path;

use anyhow::Result;
use uor_ontology::Ontology;

use extractor::{
    concept_breadcrumbs, home_breadcrumbs, namespace_breadcrumbs, namespace_summaries,
    namespaces_index_breadcrumbs, simple_breadcrumbs,
};
use nav::{build_nav, render_nav};
use renderer::{
    render_concept_page_body, render_concepts_index, render_download_page, render_explore,
    render_homepage, render_identities_page, render_namespace_page, render_namespaces_index,
    render_page, render_pipeline_page, render_search_page, render_sitemap,
};

const BASE_URL: &str = "https://uor.foundation";

/// Generates the complete website into `out_dir`.
///
/// Reads `PUBLIC_BASE_PATH` from the environment (default: `""`). Set it to
/// `/UOR-Framework` when deploying to GitHub Pages at a subdirectory.
///
/// # Errors
///
/// Returns an error if any file cannot be written.
pub fn generate(out_dir: &Path) -> Result<()> {
    let base_path = std::env::var("PUBLIC_BASE_PATH").unwrap_or_default();
    let base_path = base_path.trim_end_matches('/');

    let nav = build_nav(base_path);
    let summaries = namespace_summaries(base_path);
    let ontology = Ontology::full();

    // Track all pages for sitemap
    let mut sitemap_paths: Vec<String> = Vec::new();

    // Homepage
    let home_body = render_homepage(&summaries, base_path);
    let home_nav = render_nav(&nav, &format!("{}/", base_path));
    let home_html = render_page(
        "UOR Foundation",
        &home_body,
        &home_nav,
        &home_breadcrumbs(base_path),
        base_path,
    );
    writer::write(&out_dir.join("index.html"), &home_html)?;
    sitemap_paths.push("/".to_string());

    // Search page
    let search_body = render_search_page(base_path);
    let search_nav = render_nav(&nav, &format!("{}/search.html", base_path));
    let search_crumbs = simple_breadcrumbs("Search", base_path);
    let search_html = render_page(
        "Search",
        &search_body,
        &search_nav,
        &search_crumbs,
        base_path,
    );
    writer::write(&out_dir.join("search.html"), &search_html)?;
    sitemap_paths.push("/search.html".to_string());

    // Namespaces index page
    let ns_index_nav = render_nav(&nav, &format!("{}/namespaces/", base_path));
    let ns_index_body = render_namespaces_index(&summaries);
    let ns_index_html = render_page(
        "Namespaces",
        &ns_index_body,
        &ns_index_nav,
        &namespaces_index_breadcrumbs(base_path),
        base_path,
    );
    writer::write(
        &out_dir.join("namespaces").join("index.html"),
        &ns_index_html,
    )?;
    sitemap_paths.push("/namespaces/".to_string());

    // Namespace pages (100% auto-generated from spec)
    for module in &ontology.namespaces {
        let prefix = module.namespace.prefix;
        let page_path = format!("/namespaces/{prefix}/");
        let page_nav = render_nav(&nav, &format!("{}{}", base_path, page_path));
        let ns_breadcrumbs = namespace_breadcrumbs(module.namespace.label, base_path);
        let body = render_namespace_page(module, Some(base_path));
        let html = render_page(
            module.namespace.label,
            &body,
            &page_nav,
            &ns_breadcrumbs,
            base_path,
        );

        let out_path = out_dir.join("namespaces").join(prefix).join("index.html");
        writer::write(&out_path, &html)?;
        sitemap_paths.push(page_path);
    }

    // Pipeline page
    let pipeline_body = render_pipeline_page(&summaries, base_path);
    let pipeline_nav = render_nav(&nav, &format!("{}/pipeline/", base_path));
    let pipeline_html = render_page(
        "Pipeline",
        &pipeline_body,
        &pipeline_nav,
        &simple_breadcrumbs("Pipeline", base_path),
        base_path,
    );
    writer::write(&out_dir.join("pipeline").join("index.html"), &pipeline_html)?;
    sitemap_paths.push("/pipeline/".to_string());

    // Explore page
    let explore_body = render_explore(ontology, &summaries, base_path);
    let explore_nav = render_nav(&nav, &format!("{}/explore/", base_path));
    let explore_html = render_page(
        "Explore",
        &explore_body,
        &explore_nav,
        &simple_breadcrumbs("Explore", base_path),
        base_path,
    );
    writer::write(&out_dir.join("explore").join("index.html"), &explore_html)?;
    sitemap_paths.push("/explore/".to_string());

    // Identities page
    let identities_body = render_identities_page(ontology, base_path);
    let identities_nav = render_nav(&nav, &format!("{}/identities/", base_path));
    let identities_html = render_page(
        "Identities",
        &identities_body,
        &identities_nav,
        &simple_breadcrumbs("Identities", base_path),
        base_path,
    );
    writer::write(
        &out_dir.join("identities").join("index.html"),
        &identities_html,
    )?;
    sitemap_paths.push("/identities/".to_string());

    // Download page
    let download_body = render_download_page(base_path);
    let download_nav = render_nav(&nav, &format!("{}/download/", base_path));
    let download_html = render_page(
        "Download",
        &download_body,
        &download_nav,
        &simple_breadcrumbs("Download", base_path),
        base_path,
    );
    writer::write(&out_dir.join("download").join("index.html"), &download_html)?;
    sitemap_paths.push("/download/".to_string());

    // Concepts index + dynamically discovered concept pages
    let content_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("content");
    let concept_list = concepts::concept_page_list(&content_dir, base_path)?;
    let concepts_body = render_concepts_index(&concept_list, base_path);
    let concepts_nav = render_nav(&nav, &format!("{}/concepts/", base_path));
    let concepts_html = render_page(
        "Concepts",
        &concepts_body,
        &concepts_nav,
        &simple_breadcrumbs("Concepts", base_path),
        base_path,
    );
    writer::write(&out_dir.join("concepts").join("index.html"), &concepts_html)?;
    sitemap_paths.push("/concepts/".to_string());

    for concept in &concept_list {
        let content_path = content_dir
            .join("concepts")
            .join(format!("{}.md", concept.slug));
        let content_html = concepts::render_concept_from_file(&content_path)?;
        let extra_svg = pipeline::CONCEPT_SVG_HOOKS
            .iter()
            .find(|(slug, _)| *slug == concept.slug.as_str())
            .map(|(_, f)| f(ontology));
        let body = render_concept_page_body(&concept.title, &content_html, extra_svg.as_deref());
        let concept_nav = render_nav(&nav, &concept.url);
        let concept_crumbs = concept_breadcrumbs(&concept.title, base_path);
        let concept_html = render_page(
            &concept.title,
            &body,
            &concept_nav,
            &concept_crumbs,
            base_path,
        );
        let out_path = out_dir
            .join("concepts")
            .join(format!("{}.html", concept.slug));
        writer::write(&out_path, &concept_html)?;
        sitemap_paths.push(format!("/concepts/{}.html", concept.slug));
    }

    // Search index
    let search_index_json = search::generate_search_index(base_path)?;
    writer::write(&out_dir.join("search-index.json"), &search_index_json)?;

    // Sitemap
    let sitemap_xml = render_sitemap(BASE_URL, &sitemap_paths);
    writer::write(&out_dir.join("sitemap.xml"), &sitemap_xml)?;

    // CSS
    writer::write(&out_dir.join("css").join("style.css"), style_css())?;

    // JavaScript
    writer::write(
        &out_dir.join("js").join("search.js"),
        &search::search_js(base_path),
    )?;

    Ok(())
}

/// Returns the complete CSS stylesheet.
fn style_css() -> &'static str {
    include_str!("../static/css/style.css")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_index_has_all_classes() {
        let entries = extractor::build_search_index("");
        let class_count = entries.iter().filter(|e| e.kind == "class").count();
        assert_eq!(
            class_count,
            uor_ontology::counts::CLASSES,
            "Expected {} class entries in search index",
            uor_ontology::counts::CLASSES
        );
    }

    #[test]
    fn namespace_summaries_count() {
        let summaries = namespace_summaries("");
        assert_eq!(summaries.len(), uor_ontology::counts::NAMESPACES);
    }

    #[test]
    fn nav_renders_non_empty() {
        let nav = build_nav("");
        let html = render_nav(&nav, "/");
        assert!(!html.is_empty());
        assert!(html.contains("UOR Foundation") || html.contains("Home"));
    }
}
