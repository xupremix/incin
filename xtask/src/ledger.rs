//! Validates the canonical execution ledger (`GOV-003`).
//!
//! The ledger exists twice on purpose: as a human-readable table in
//! `PROPOSALS.md` and as a machine-readable mirror in `docs/plan/ledger.toml`.
//! Two copies of anything drift, so this task makes drift a build failure
//! rather than something a reader discovers months later.
//!
//! Checks performed:
//!
//! 1. every task in the table appears in the mirror, and vice versa;
//! 2. tier, theme, status, and dependencies agree between the two;
//! 3. every dependency names a task that exists;
//! 4. the dependency graph is acyclic;
//! 5. no task depends on a less-mature tier (Appendix B, rule 1);
//! 6. every task marked complete records concrete evidence.

use std::collections::{BTreeMap, BTreeSet};
use std::process::ExitCode;

const PROPOSALS: &str = "PROPOSALS.md";
const MIRROR: &str = "docs/plan/ledger.toml";

/// One ledger entry, as read from either representation.
#[derive(Debug, PartialEq, Eq)]
struct Task {
    tier: String,
    theme: String,
    status: String,
    deps: Vec<String>,
}

/// Maturity rank. A task may only depend on a tier at least as mature as its
/// own, so `core` (0) cannot depend on `preview` (1) or `exploratory` (2).
fn maturity(tier: &str) -> Option<u8> {
    match tier {
        "core" => Some(0),
        "preview" => Some(1),
        "exploratory" => Some(2),
        _ => None,
    }
}

fn status_word(marker: &str) -> &'static str {
    match marker {
        "x" => "complete",
        "~" => "active",
        "!" => "blocked",
        "-" => "deferred",
        _ => "planned",
    }
}

/// Parses the markdown ledger table out of `PROPOSALS.md`.
///
/// Rows look like `| SHP-002 | core | shape | [ ] | SHP-001 | ... |`, and the
/// em dash stands in for "no dependencies".
fn parse_table(src: &str) -> BTreeMap<String, Task> {
    let mut out = BTreeMap::new();
    for line in src.lines() {
        let line = line.trim();
        if !line.starts_with('|') {
            continue;
        }
        let cells: Vec<&str> = line.split('|').map(str::trim).collect();
        // cells[0] is empty (leading pipe); id, tier, theme, status, deps follow.
        if cells.len() < 6 {
            continue;
        }
        // Task ids are a two- or three-letter uppercase prefix, a dash, and
        // three digits: `UX-001`, `SHP-002`.
        let id = cells[1];
        let looks_like_id = match id.split_once('-') {
            Some((prefix, number)) => {
                (2..=3).contains(&prefix.len())
                    && prefix.chars().all(|c| c.is_ascii_uppercase())
                    && number.len() == 3
                    && number.chars().all(|c| c.is_ascii_digit())
            }
            None => false,
        };
        if !looks_like_id {
            continue;
        }
        let status = cells[4];
        let marker = status
            .strip_prefix('[')
            .and_then(|s| s.strip_suffix(']'))
            .unwrap_or(status)
            .trim();
        let deps = if cells[5] == "—" || cells[5].is_empty() {
            Vec::new()
        } else {
            cells[5].split(',').map(|d| d.trim().to_string()).collect()
        };
        out.insert(
            id.to_string(),
            Task {
                tier: cells[2].to_string(),
                theme: cells[3].to_string(),
                status: status_word(marker).to_string(),
                deps,
            },
        );
    }
    out
}

/// Parses the TOML mirror, returning the tasks and the set of ids whose
/// `completed_evidence` field is non-empty.
fn parse_mirror(src: &str) -> Result<(BTreeMap<String, Task>, BTreeSet<String>), String> {
    // `toml` 1.x routes `FromStr` for `Value` through a different parser than
    // `from_str` and rejects a whole document there; `from_str` is the spelling
    // that still deserializes one.
    let doc: toml::Value = toml::from_str(src).map_err(|e| format!("{MIRROR}: {e}"))?;
    let entries = doc
        .get("task")
        .and_then(toml::Value::as_array)
        .ok_or_else(|| format!("{MIRROR}: missing `[[task]]` array"))?;

    let mut out = BTreeMap::new();
    let mut with_evidence = BTreeSet::new();
    for entry in entries {
        let get = |k: &str| -> Result<String, String> {
            entry
                .get(k)
                .and_then(toml::Value::as_str)
                .map(str::to_string)
                .ok_or_else(|| format!("{MIRROR}: a task is missing the `{k}` field"))
        };
        let id = get("id")?;
        let deps = entry
            .get("deps")
            .and_then(toml::Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(toml::Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        if entry
            .get("completed_evidence")
            .and_then(toml::Value::as_str)
            .is_some_and(|s| !s.trim().is_empty())
        {
            with_evidence.insert(id.clone());
        }
        out.insert(
            id,
            Task {
                tier: get("tier")?,
                theme: get("theme")?,
                status: get("status")?,
                deps,
            },
        );
    }
    Ok((out, with_evidence))
}

/// Reports every cycle-forming edge via an iterative depth-first search.
fn find_cycles(tasks: &BTreeMap<String, Task>, errors: &mut Vec<String>) {
    #[derive(Clone, Copy, PartialEq)]
    enum Mark {
        Open,
        Done,
    }
    let mut marks: BTreeMap<&str, Mark> = BTreeMap::new();

    for root in tasks.keys() {
        if marks.contains_key(root.as_str()) {
            continue;
        }
        // (node, index of next dependency to visit)
        let mut stack: Vec<(&str, usize)> = vec![(root, 0)];
        marks.insert(root, Mark::Open);
        while let Some((node, index)) = stack.pop() {
            let Some(task) = tasks.get(node) else {
                continue;
            };
            if index < task.deps.len() {
                stack.push((node, index + 1));
                let dep = task.deps[index].as_str();
                if !tasks.contains_key(dep) {
                    continue; // reported separately as an unknown dependency
                }
                match marks.get(dep) {
                    Some(Mark::Open) => {
                        errors.push(format!("cycle: {node} -> {dep}"));
                    }
                    Some(Mark::Done) => {}
                    None => {
                        marks.insert(dep, Mark::Open);
                        stack.push((dep, 0));
                    }
                }
            } else {
                marks.insert(node, Mark::Done);
            }
        }
    }
}

pub fn check() -> ExitCode {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives one level below the workspace root")
        .to_path_buf();

    let proposals = match std::fs::read_to_string(root.join(PROPOSALS)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read {PROPOSALS}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mirror_src = match std::fs::read_to_string(root.join(MIRROR)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read {MIRROR}: {e}");
            return ExitCode::FAILURE;
        }
    };

    let table = parse_table(&proposals);
    let (mirror, with_evidence) = match parse_mirror(&mirror_src) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mut errors = Vec::new();

    if table.is_empty() {
        errors.push(format!("{PROPOSALS}: no ledger rows found"));
    }

    // 1 and 2: the two representations must agree.
    for (id, row) in &table {
        match mirror.get(id) {
            None => errors.push(format!("{id}: in {PROPOSALS} but not in {MIRROR}")),
            Some(entry) if entry != row => errors.push(format!(
                "{id}: {PROPOSALS} and {MIRROR} disagree\n      table:  {row:?}\n      mirror: {entry:?}"
            )),
            Some(_) => {}
        }
    }
    for id in mirror.keys() {
        if !table.contains_key(id) {
            errors.push(format!("{id}: in {MIRROR} but not in {PROPOSALS}"));
        }
    }

    for (id, task) in &table {
        // 3: dependencies must exist.
        for dep in &task.deps {
            if !table.contains_key(dep) {
                errors.push(format!("{id}: depends on unknown task {dep}"));
                continue;
            }
            // 5: tier maturity ordering.
            let (Some(own), Some(dep_tier)) = (
                maturity(&task.tier),
                table.get(dep).and_then(|d| maturity(&d.tier)),
            ) else {
                continue;
            };
            if dep_tier > own {
                errors.push(format!(
                    "{id} ({}) depends on {dep} ({}); a task may not depend on a less-mature tier",
                    task.tier, table[dep].tier
                ));
            }
        }
        if maturity(&task.tier).is_none() {
            errors.push(format!("{id}: unknown tier `{}`", task.tier));
        }
        // 6: completion requires evidence.
        if task.status == "complete" && !with_evidence.contains(id) {
            errors.push(format!(
                "{id}: marked complete but records no `completed_evidence` in {MIRROR}"
            ));
        }
    }

    // 4: the graph must be acyclic.
    find_cycles(&table, &mut errors);

    if errors.is_empty() {
        let complete = table.values().filter(|t| t.status == "complete").count();
        println!("ledger ok: {} tasks, {complete} complete", table.len());
        ExitCode::SUCCESS
    } else {
        eprintln!("ledger validation failed with {} problem(s):", errors.len());
        for e in &errors {
            eprintln!("  - {e}");
        }
        ExitCode::FAILURE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TABLE: &str = "\
| ID | Tier | Theme | Status | Dependencies | Target | Deliverable | Evidence |
|---|---|---|---|---|---|---|---|
| GOV-001 | core | gov | [x] | — | `a` | d | `e` |
| SHP-001 | core | shape | [ ] | GOV-001 | `a` | d | `e` |
";

    #[test]
    fn parses_ids_dependencies_and_status() {
        let t = parse_table(TABLE);
        assert_eq!(t.len(), 2);
        assert_eq!(t["GOV-001"].status, "complete");
        assert!(
            t["GOV-001"].deps.is_empty(),
            "em dash means no dependencies"
        );
        assert_eq!(t["SHP-001"].deps, vec!["GOV-001"]);
        assert_eq!(t["SHP-001"].theme, "shape");
    }

    #[test]
    fn ignores_unrelated_markdown_tables() {
        let t = parse_table("| Tier | Count |\n|---|---|\n| Core | 39 |\n");
        assert!(t.is_empty(), "rows without a task id must not be parsed");
    }

    #[test]
    fn maturity_forbids_core_depending_on_preview() {
        assert!(maturity("core") < maturity("preview"));
        assert!(maturity("preview") < maturity("exploratory"));
        assert_eq!(maturity("nonsense"), None);
    }

    #[test]
    fn detects_a_cycle() {
        let mut tasks = BTreeMap::new();
        for (id, dep) in [("AAA-001", "AAA-002"), ("AAA-002", "AAA-001")] {
            tasks.insert(
                id.to_string(),
                Task {
                    tier: "core".into(),
                    theme: "gov".into(),
                    status: "planned".into(),
                    deps: vec![dep.to_string()],
                },
            );
        }
        let mut errors = Vec::new();
        find_cycles(&tasks, &mut errors);
        assert!(!errors.is_empty(), "a two-node cycle must be reported");
    }

    #[test]
    fn acyclic_graph_reports_nothing() {
        let t = parse_table(TABLE);
        let mut errors = Vec::new();
        find_cycles(&t, &mut errors);
        assert!(errors.is_empty(), "{errors:?}");
    }
}
