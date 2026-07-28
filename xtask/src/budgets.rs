//! Deterministic regression-budget and Cargo-feature inventory gate (GOV-005).

use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, ExitCode};

const CONFIG_PATH: &str = "docs/plan/budgets.toml";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BudgetConfig {
    schema: u32,
    baseline: String,
    #[serde(default)]
    runtime: Vec<RuntimeBudget>,
    #[serde(default)]
    artifact: Vec<ArtifactBudget>,
    #[serde(default)]
    feature_crate: Vec<FeatureCrate>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeBudget {
    id: String,
    backend: String,
    baseline_high_ns: f64,
    max_high_ns: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactBudget {
    profile: String,
    metric: String,
    baseline_bytes: u64,
    max_bytes: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeatureCrate {
    package: String,
    manifest: String,
    default: Vec<String>,
    #[serde(default)]
    feature: Vec<Feature>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Feature {
    name: String,
    enables: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct Baseline {
    schema: u32,
    #[serde(default)]
    compile: BTreeMap<String, BTreeMap<String, toml::Value>>,
    #[serde(default)]
    runtime: Vec<BaselineRuntime>,
}

#[derive(Debug, Deserialize)]
struct BaselineRuntime {
    id: String,
    backend: String,
    high_ns: f64,
}

pub fn check() -> ExitCode {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask must be inside the workspace")
        .to_owned();

    match check_root(&root) {
        Ok(summary) => {
            println!(
                "budgets ok: {} runtime, {} artifacts, {} feature crates, {} features",
                summary.runtime, summary.artifacts, summary.feature_crates, summary.features
            );
            ExitCode::SUCCESS
        }
        Err(errors) => {
            for error in &errors {
                eprintln!("budget error: {error}");
            }
            eprintln!("budgets failed: {} error(s)", errors.len());
            ExitCode::FAILURE
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Summary {
    runtime: usize,
    artifacts: usize,
    feature_crates: usize,
    features: usize,
}

fn check_root(root: &Path) -> Result<Summary, Vec<String>> {
    let config_text = read(root.join(CONFIG_PATH))?;
    let config: BudgetConfig = parse(CONFIG_PATH, &config_text)?;
    if !safe_relative(&config.baseline) {
        return Err(vec![format!(
            "baseline path must be a safe workspace-relative path: {}",
            config.baseline
        )]);
    }
    let baseline_text = read(root.join(&config.baseline))?;
    let baseline: Baseline = parse(&config.baseline, &baseline_text)?;

    let mut errors = validate_numeric(&config, &baseline);
    errors.extend(validate_features(&config, |manifest| {
        read(root.join(manifest)).map_err(|errors| errors.join("; "))
    }));
    match metadata_feature_manifests(root) {
        Ok(manifests) => errors.extend(validate_inventory_coverage(&config, &manifests)),
        Err(error) => errors.push(error),
    }

    if errors.is_empty() {
        Ok(Summary {
            runtime: config.runtime.len(),
            artifacts: config.artifact.len(),
            feature_crates: config.feature_crate.len(),
            features: config
                .feature_crate
                .iter()
                .map(|item| item.feature.len())
                .sum(),
        })
    } else {
        Err(errors)
    }
}

fn read(path: PathBuf) -> Result<String, Vec<String>> {
    fs::read_to_string(&path).map_err(|error| vec![format!("{}: {error}", path.display())])
}

fn parse<T: for<'de> Deserialize<'de>>(name: &str, text: &str) -> Result<T, Vec<String>> {
    toml::from_str(text).map_err(|error| vec![format!("{name}: {error}")])
}

fn safe_relative(path: &str) -> bool {
    let path = Path::new(path);
    !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

fn validate_numeric(config: &BudgetConfig, baseline: &Baseline) -> Vec<String> {
    let mut errors = Vec::new();
    if config.schema != 1 {
        errors.push(format!(
            "unsupported budget schema {} (expected 1)",
            config.schema
        ));
    }
    if baseline.schema != 1 {
        errors.push(format!(
            "unsupported baseline schema {} (expected 1)",
            baseline.schema
        ));
    }

    let mut actual_runtime = BTreeMap::new();
    for series in &baseline.runtime {
        let key = (series.backend.as_str(), series.id.as_str());
        if actual_runtime.insert(key, series.high_ns).is_some() {
            errors.push(format!(
                "duplicate baseline runtime series {}/{}",
                series.backend, series.id
            ));
        }
    }

    let mut budget_runtime = BTreeSet::new();
    for budget in &config.runtime {
        let key = (budget.backend.as_str(), budget.id.as_str());
        if !budget_runtime.insert(key) {
            errors.push(format!(
                "duplicate runtime budget {}/{}",
                budget.backend, budget.id
            ));
            continue;
        }
        match actual_runtime.get(&key) {
            None => errors.push(format!(
                "runtime budget {}/{} has no baseline series",
                budget.backend, budget.id
            )),
            Some(actual) => {
                validate_float_budget(
                    &format!("runtime {}/{}", budget.backend, budget.id),
                    *actual,
                    budget.baseline_high_ns,
                    budget.max_high_ns,
                    &mut errors,
                );
            }
        }
    }
    for key in actual_runtime.keys() {
        if !budget_runtime.contains(key) {
            errors.push(format!(
                "baseline runtime series {}/{} has no budget",
                key.0, key.1
            ));
        }
    }

    let mut actual_artifacts = BTreeMap::new();
    for (profile, values) in &baseline.compile {
        for (metric, value) in values {
            if !metric.ends_with("_bytes") {
                continue;
            }
            match value
                .as_integer()
                .and_then(|value| u64::try_from(value).ok())
            {
                Some(value) => {
                    actual_artifacts.insert((profile.as_str(), metric.as_str()), value);
                }
                None => errors.push(format!(
                    "compile artifact {profile}/{metric} must be a non-negative integer"
                )),
            }
        }
    }

    let mut budget_artifacts = BTreeSet::new();
    for budget in &config.artifact {
        let key = (budget.profile.as_str(), budget.metric.as_str());
        if !budget_artifacts.insert(key) {
            errors.push(format!(
                "duplicate artifact budget {}/{}",
                budget.profile, budget.metric
            ));
            continue;
        }
        match actual_artifacts.get(&key) {
            None => errors.push(format!(
                "artifact budget {}/{} has no baseline metric",
                budget.profile, budget.metric
            )),
            Some(actual) => {
                if *actual != budget.baseline_bytes {
                    errors.push(format!(
                        "artifact {}/{} baseline drifted: config {}, baseline {}",
                        budget.profile, budget.metric, budget.baseline_bytes, actual
                    ));
                }
                if budget.max_bytes < budget.baseline_bytes {
                    errors.push(format!(
                        "artifact {}/{} maximum {} is below its declared baseline {}",
                        budget.profile, budget.metric, budget.max_bytes, budget.baseline_bytes
                    ));
                }
                if *actual > budget.max_bytes {
                    errors.push(format!(
                        "artifact {}/{} exceeds budget: {} > {} bytes",
                        budget.profile, budget.metric, actual, budget.max_bytes
                    ));
                }
            }
        }
    }
    for key in actual_artifacts.keys() {
        if !budget_artifacts.contains(key) {
            errors.push(format!(
                "baseline artifact {}/{} has no budget",
                key.0, key.1
            ));
        }
    }
    errors
}

fn validate_float_budget(
    name: &str,
    actual: f64,
    declared: f64,
    maximum: f64,
    errors: &mut Vec<String>,
) {
    if !actual.is_finite() || actual < 0.0 || !declared.is_finite() || declared < 0.0 {
        errors.push(format!("{name} baseline must be finite and non-negative"));
        return;
    }
    if !maximum.is_finite() || maximum < declared {
        errors.push(format!(
            "{name} maximum {maximum} is invalid or below its declared baseline {declared}"
        ));
    }
    let tolerance = declared.abs().mul_add(1e-12, 1e-9);
    if (actual - declared).abs() > tolerance {
        errors.push(format!(
            "{name} baseline drifted: config {declared}, baseline {actual}"
        ));
    }
    if actual > maximum {
        errors.push(format!("{name} exceeds budget: {actual} > {maximum} ns"));
    }
}

#[derive(Debug, Deserialize)]
struct CargoMetadata {
    packages: Vec<CargoPackage>,
    workspace_members: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct CargoPackage {
    id: String,
    manifest_path: String,
    features: BTreeMap<String, Vec<String>>,
}

fn metadata_feature_manifests(root: &Path) -> Result<BTreeSet<String>, String> {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());
    let output = Command::new(cargo)
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .current_dir(root)
        .output()
        .map_err(|error| format!("failed to run cargo metadata: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let metadata: CargoMetadata = serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("cargo metadata returned invalid JSON: {error}"))?;
    let members: BTreeSet<_> = metadata
        .workspace_members
        .iter()
        .map(String::as_str)
        .collect();
    let mut manifests = BTreeSet::new();
    for package in metadata.packages {
        if !members.contains(package.id.as_str()) || package.features.is_empty() {
            continue;
        }
        let relative = Path::new(&package.manifest_path)
            .strip_prefix(root)
            .map_err(|_| {
                format!(
                    "workspace manifest is outside the root: {}",
                    package.manifest_path
                )
            })?;
        let relative = relative.to_str().ok_or_else(|| {
            format!(
                "workspace manifest path is not UTF-8: {}",
                relative.display()
            )
        })?;
        manifests.insert(relative.to_owned());
    }
    Ok(manifests)
}

fn validate_inventory_coverage(config: &BudgetConfig, actual: &BTreeSet<String>) -> Vec<String> {
    let declared: BTreeSet<_> = config
        .feature_crate
        .iter()
        .map(|item| item.manifest.clone())
        .collect();
    let mut errors = Vec::new();
    for missing in actual.difference(&declared) {
        errors.push(format!(
            "feature-bearing workspace manifest {missing} is missing from the inventory"
        ));
    }
    for stale in declared.difference(actual) {
        errors.push(format!(
            "inventoried manifest {stale} is not a feature-bearing workspace member"
        ));
    }
    errors
}

fn validate_features<F>(config: &BudgetConfig, mut load: F) -> Vec<String>
where
    F: FnMut(&str) -> Result<String, String>,
{
    let mut errors = Vec::new();
    let mut manifests = BTreeSet::new();
    let mut packages = BTreeSet::new();
    for inventory in &config.feature_crate {
        if !safe_relative(&inventory.manifest) {
            errors.push(format!(
                "feature manifest must be a safe workspace-relative path: {}",
                inventory.manifest
            ));
            continue;
        }
        if !manifests.insert(inventory.manifest.as_str()) {
            errors.push(format!("duplicate feature manifest {}", inventory.manifest));
        }
        if !packages.insert(inventory.package.as_str()) {
            errors.push(format!("duplicate feature package {}", inventory.package));
        }
        match load(&inventory.manifest) {
            Ok(text) => errors.extend(validate_feature_manifest(inventory, &text)),
            Err(error) => errors.push(error),
        }
    }
    errors
}

fn validate_feature_manifest(inventory: &FeatureCrate, text: &str) -> Vec<String> {
    let mut errors = Vec::new();
    let value: toml::Value = match toml::from_str(text) {
        Ok(value) => value,
        Err(error) => return vec![format!("{}: {error}", inventory.manifest)],
    };
    let package = value
        .get("package")
        .and_then(|value| value.get("name"))
        .and_then(toml::Value::as_str);
    if package != Some(inventory.package.as_str()) {
        errors.push(format!(
            "{} package mismatch: expected {}, found {}",
            inventory.manifest,
            inventory.package,
            package.unwrap_or("<missing>")
        ));
    }
    let Some(features) = value.get("features").and_then(toml::Value::as_table) else {
        errors.push(format!("{} has no [features] table", inventory.manifest));
        return errors;
    };

    let actual_default = feature_values(features.get("default"), "default", &mut errors);
    compare_values(
        &inventory.manifest,
        "default",
        &inventory.default,
        &actual_default,
        &mut errors,
    );

    let declared: BTreeMap<_, _> = inventory
        .feature
        .iter()
        .map(|feature| (feature.name.as_str(), feature))
        .collect();
    if declared.len() != inventory.feature.len() {
        errors.push(format!(
            "{} contains duplicate inventory features",
            inventory.manifest
        ));
    }
    let actual_names: BTreeSet<_> = features
        .keys()
        .filter(|name| name.as_str() != "default")
        .map(String::as_str)
        .collect();
    let declared_names: BTreeSet<_> = declared.keys().copied().collect();
    for missing in actual_names.difference(&declared_names) {
        errors.push(format!(
            "{} feature `{missing}` is missing from the inventory",
            inventory.manifest
        ));
    }
    for removed in declared_names.difference(&actual_names) {
        errors.push(format!(
            "{} inventories absent feature `{removed}`",
            inventory.manifest
        ));
    }
    for (name, expected) in declared {
        let actual = feature_values(features.get(name), name, &mut errors);
        compare_values(
            &inventory.manifest,
            name,
            &expected.enables,
            &actual,
            &mut errors,
        );
    }
    errors
}

fn feature_values(
    value: Option<&toml::Value>,
    name: &str,
    errors: &mut Vec<String>,
) -> Vec<String> {
    let Some(values) = value.and_then(toml::Value::as_array) else {
        errors.push(format!("feature `{name}` must be an array"));
        return Vec::new();
    };
    values
        .iter()
        .filter_map(|value| match value.as_str() {
            Some(value) => Some(value.to_owned()),
            None => {
                errors.push(format!("feature `{name}` contains a non-string value"));
                None
            }
        })
        .collect()
}

fn compare_values(
    manifest: &str,
    feature: &str,
    expected: &[String],
    actual: &[String],
    errors: &mut Vec<String>,
) {
    let expected: BTreeSet<_> = expected.iter().map(String::as_str).collect();
    let actual: BTreeSet<_> = actual.iter().map(String::as_str).collect();
    if expected != actual {
        errors.push(format!(
            "{manifest} feature `{feature}` drifted: expected {expected:?}, found {actual:?}"
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = r#"
schema = 1
baseline = "baseline.toml"
[[runtime]]
id = "add"
backend = "cpu"
baseline_high_ns = 10.0
max_high_ns = 11.0
[[artifact]]
profile = "cpu"
metric = "binary_bytes"
baseline_bytes = 100
max_bytes = 120
[[feature_crate]]
package = "demo"
manifest = "demo/Cargo.toml"
default = ["std"]
[[feature_crate.feature]]
name = "std"
enables = []
"#;

    const BASELINE: &str = r#"
schema = 1
[compile.cpu]
binary_bytes = 100
[[runtime]]
id = "add"
backend = "cpu"
high_ns = 10.0
"#;

    const MANIFEST: &str = r#"
[package]
name = "demo"
[features]
default = ["std"]
std = []
"#;

    fn fixtures() -> (BudgetConfig, Baseline) {
        (
            toml::from_str(CONFIG).unwrap(),
            toml::from_str(BASELINE).unwrap(),
        )
    }

    #[test]
    fn valid_documents_pass() {
        let (config, baseline) = fixtures();
        assert!(validate_numeric(&config, &baseline).is_empty());
        assert!(validate_features(&config, |_| Ok(MANIFEST.into())).is_empty());
    }

    #[test]
    fn exceeded_runtime_budget_fails() {
        let (config, mut baseline) = fixtures();
        baseline.runtime[0].high_ns = 12.0;
        let errors = validate_numeric(&config, &baseline);
        assert!(errors.iter().any(|error| error.contains("exceeds budget")));
    }

    #[test]
    fn missing_runtime_budget_fails() {
        let (mut config, baseline) = fixtures();
        config.runtime.clear();
        let errors = validate_numeric(&config, &baseline);
        assert!(errors.iter().any(|error| error.contains("has no budget")));
    }

    #[test]
    fn duplicate_artifact_budget_fails() {
        let (mut config, baseline) = fixtures();
        let duplicate: ArtifactBudget = toml::from_str(
            "profile = 'cpu'\nmetric = 'binary_bytes'\nbaseline_bytes = 100\nmax_bytes = 120",
        )
        .unwrap();
        config.artifact.push(duplicate);
        let errors = validate_numeric(&config, &baseline);
        assert!(
            errors
                .iter()
                .any(|error| error.contains("duplicate artifact"))
        );
    }

    #[test]
    fn feature_drift_fails() {
        let (config, _) = fixtures();
        let changed = MANIFEST.replace("std = []", "std = []\nwgpu = []");
        let errors = validate_features(&config, |_| Ok(changed.clone()));
        assert!(
            errors
                .iter()
                .any(|error| error.contains("missing from the inventory"))
        );
    }

    #[test]
    fn uninventoried_feature_crate_fails() {
        let (config, _) = fixtures();
        let actual = BTreeSet::from(["demo/Cargo.toml".to_owned(), "new/Cargo.toml".to_owned()]);
        let errors = validate_inventory_coverage(&config, &actual);
        assert!(errors.iter().any(|error| error.contains("new/Cargo.toml")));
    }

    #[test]
    fn parent_paths_are_rejected() {
        assert!(!safe_relative("../baseline.toml"));
        assert!(!safe_relative("/tmp/baseline.toml"));
        assert!(safe_relative("docs/plan/baseline.toml"));
    }
}
