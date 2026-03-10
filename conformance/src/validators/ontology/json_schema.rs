//! JSON Schema artifact validator.
//!
//! Validates that the generated `uor.foundation.schema.json` file is
//! well-formed and contains the expected schema structure.

use std::path::Path;

use anyhow::{Context, Result};

use crate::report::{ConformanceReport, TestResult};

/// Validates the JSON Schema artifact for structural correctness.
///
/// Checks that `uor.foundation.schema.json` exists, parses as valid JSON,
/// contains the expected `$schema`, `$defs`, class count, and enum class
/// entries.
///
/// # Errors
///
/// Returns an error if the artifact file cannot be read.
pub fn validate(artifacts: &Path) -> Result<ConformanceReport> {
    let mut report = ConformanceReport::new();
    let validator = "ontology/json_schema";

    let schema_path = artifacts.join("uor.foundation.schema.json");
    if !schema_path.exists() {
        report.push(TestResult::fail(
            validator,
            "uor.foundation.schema.json not found in artifacts directory",
        ));
        return Ok(report);
    }

    let content = std::fs::read_to_string(&schema_path)
        .with_context(|| format!("Failed to read {}", schema_path.display()))?;

    let mut issues: Vec<String> = Vec::new();

    // Parse JSON
    let json: serde_json::Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(e) => {
            issues.push(format!("Invalid JSON: {e}"));
            report.push(TestResult::fail_with_details(
                validator,
                "uor.foundation.schema.json has structural issues",
                issues,
            ));
            return Ok(report);
        }
    };

    // $schema key
    if json.get("$schema").and_then(|v| v.as_str())
        != Some("https://json-schema.org/draft/2020-12/schema")
    {
        issues.push("Missing or incorrect $schema value".to_string());
    }

    // $defs
    let ontology = uor_ontology::Ontology::full();
    let expected_count = ontology.class_count();

    if let Some(defs) = json.get("$defs").and_then(|v| v.as_object()) {
        if defs.len() != expected_count {
            issues.push(format!(
                "$defs has {} entries, expected {}",
                defs.len(),
                expected_count
            ));
        }

        // Enum classes must have "enum" key (keys are qualified: "prefix/Name")
        for name in uor_ontology::Ontology::enum_class_names() {
            let suffix = format!("/{}", name);
            let found = defs.iter().find(|(k, _)| k.ends_with(&suffix));
            if let Some((_, entry)) = found {
                if entry.get("enum").is_none() {
                    issues.push(format!("Enum class '{}' missing 'enum' keyword", name));
                }
            } else {
                issues.push(format!("Missing $defs entry for enum class '{}'", name));
            }
        }
    } else {
        issues.push("Missing $defs object".to_string());
    }

    // Version in description
    if let Some(desc) = json.get("description").and_then(|v| v.as_str()) {
        if !desc.contains(ontology.version) {
            issues.push(format!(
                "Version '{}' not found in description",
                ontology.version
            ));
        }
    } else {
        issues.push("Missing description field".to_string());
    }

    if issues.is_empty() {
        report.push(TestResult::pass(
            validator,
            format!(
                "uor.foundation.schema.json is well-formed \
                 ({} bytes, {} $defs entries)",
                content.len(),
                expected_count
            ),
        ));
    } else {
        report.push(TestResult::fail_with_details(
            validator,
            "uor.foundation.schema.json has structural issues",
            issues,
        ));
    }

    Ok(report)
}
