//! Checking whether a newer `incin` has been published.
//!
//! This is deliberately awkward to trigger. It runs only when a person asks
//! for it by name, never as a side effect of `cargo incin build`, `check`,
//! `test`, or any other delegated command: a build tool that contacts the
//! network on its own is a tool people are right to distrust, and the check is
//! not worth that. It is also compiled out entirely unless the
//! `update-check` feature is on, so a build that never wants the capability
//! does not carry the HTTP stack that provides it.
//!
//! What it compares is the version of the **running binary**
//! (`CARGO_PKG_VERSION`, baked in at compile time) against the newest
//! non-yanked release on crates.io. That is not the same question as "is the
//! version my project depends on current", which needs manifest resolution and
//! is a separate thing entirely.
//!
//! The transport is the crates.io sparse index rather than `cargo search` or
//! `cargo info`. The index is a documented, versioned protocol returning one
//! JSON object per line; the human output of those two commands carries no
//! stability guarantee and would break silently on a cargo release.

/// The version of the `incin` this binary was built from.
pub const CURRENT: &str = env!("CARGO_PKG_VERSION");

/// How long to wait for crates.io before giving up.
///
/// Short on purpose. The check is a courtesy line in a diagnostic report, and
/// a report that hangs is worse than one that says it could not look.
#[cfg(feature = "update-check")]
const TIMEOUT: core::time::Duration = core::time::Duration::from_secs(3);

/// What a check concluded.
///
/// Every variant other than [`Available`](UpdateStatus::Available) is a
/// non-event: the caller reports it and carries on. Nothing here is ever an
/// error that should fail a command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpdateStatus {
    /// The running binary is at or ahead of the newest published release.
    Current {
        /// The version this binary was built from.
        running: String,
    },
    /// A newer release exists.
    Available {
        /// The version this binary was built from.
        running: String,
        /// The newest non-yanked version on crates.io.
        latest: String,
    },
    /// The check was not attempted because the build asked to stay offline.
    ///
    /// Set by `CARGO_NET_OFFLINE=true`, the same switch that puts cargo
    /// itself offline, so one setting covers both.
    Offline,
    /// The check was not attempted because this build has no update check
    /// compiled into it.
    NotCompiledIn,
    /// The check was attempted and did not produce an answer.
    ///
    /// Carries the reason so a person can tell a proxy problem from a parse
    /// problem, but is never fatal.
    Unknown {
        /// Why no version could be determined.
        reason: String,
    },
}

impl core::fmt::Display for UpdateStatus {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Current { running } => {
                write!(f, "incin {running} is the newest published release")
            }
            Self::Available { running, latest } => write!(
                f,
                "incin {latest} is available (running {running}); \
                 update with `cargo install cargo-incin --force`"
            ),
            Self::Offline => write!(f, "update check skipped: CARGO_NET_OFFLINE is set"),
            Self::NotCompiledIn => write!(
                f,
                "update check skipped: this build was compiled without the \
                 `update-check` feature"
            ),
            Self::Unknown { reason } => write!(f, "update check inconclusive: {reason}"),
        }
    }
}

/// Asks crates.io whether a newer `incin` exists.
///
/// Never panics and never returns an error: a network failure, a proxy, a
/// malformed response, and an offline build all come back as a variant the
/// caller can print.
#[must_use]
pub fn check() -> UpdateStatus {
    if offline() {
        return UpdateStatus::Offline;
    }
    check_online()
}

/// Whether the environment has asked to stay off the network.
fn offline() -> bool {
    std::env::var("CARGO_NET_OFFLINE")
        .map(|value| value.eq_ignore_ascii_case("true") || value == "1")
        .unwrap_or(false)
}

#[cfg(not(feature = "update-check"))]
fn check_online() -> UpdateStatus {
    UpdateStatus::NotCompiledIn
}

#[cfg(feature = "update-check")]
fn check_online() -> UpdateStatus {
    let url = format!("https://index.crates.io/{}", index_path("incin"));
    let body = match ureq::get(&url)
        .config()
        .timeout_global(Some(TIMEOUT))
        .build()
        .call()
        .and_then(|mut response| response.body_mut().read_to_string())
    {
        Ok(body) => body,
        Err(error) => {
            return UpdateStatus::Unknown {
                reason: error.to_string(),
            };
        }
    };

    match newest_release(&body) {
        Some(latest) => compare(CURRENT, &latest),
        None => UpdateStatus::Unknown {
            reason: String::from("no non-yanked release in the index response"),
        },
    }
}

/// The sparse index path for a crate name, per the registry protocol.
///
/// One, two, and three character names are special-cased by the protocol
/// itself; everything else is grouped by its first two and next two
/// characters. `incin` is five characters, so it lands at `in/ci/incin`.
#[cfg(feature = "update-check")]
fn index_path(name: &str) -> String {
    let lower = name.to_ascii_lowercase();
    match lower.len() {
        0 => String::new(),
        1 => format!("1/{lower}"),
        2 => format!("2/{lower}"),
        3 => format!("3/{}/{}", &lower[..1], lower),
        _ => format!("{}/{}/{}", &lower[..2], &lower[2..4], lower),
    }
}

/// The newest non-yanked version in a sparse index response.
///
/// The response is newline-delimited JSON, one object per published version,
/// in publication order. Yanked versions are skipped: reporting one as the
/// latest release would send a person to a version they cannot install, which
/// is worse than not checking at all.
#[cfg(feature = "update-check")]
fn newest_release(body: &str) -> Option<String> {
    let mut newest: Option<(Version, String)> = None;
    for line in body.lines().filter(|line| !line.trim().is_empty()) {
        let Ok(entry) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if entry.get("yanked").and_then(serde_json::Value::as_bool) == Some(true) {
            continue;
        }
        let Some(raw) = entry.get("vers").and_then(serde_json::Value::as_str) else {
            continue;
        };
        // Pre-releases are not offered as updates. Someone running a stable
        // build should not be pointed at an alpha, and someone deliberately
        // running a pre-release knows where to look.
        let Some(parsed) = Version::parse(raw) else {
            continue;
        };
        if newest.as_ref().is_none_or(|(best, _)| parsed > *best) {
            newest = Some((parsed, raw.to_string()));
        }
    }
    newest.map(|(_, raw)| raw)
}

/// Decides the status from two version strings.
#[cfg(feature = "update-check")]
fn compare(running: &str, latest: &str) -> UpdateStatus {
    match (Version::parse(running), Version::parse(latest)) {
        (Some(current), Some(newest)) if newest > current => UpdateStatus::Available {
            running: running.to_string(),
            latest: latest.to_string(),
        },
        (Some(_), Some(_)) => UpdateStatus::Current {
            running: running.to_string(),
        },
        _ => UpdateStatus::Unknown {
            reason: format!("could not compare `{running}` with `{latest}`"),
        },
    }
}

/// A release version, ordered the way cargo orders them.
///
/// Deliberately rejects anything carrying a pre-release or build suffix rather
/// than trying to order it. Pulling in a full semver implementation to decide
/// whether to print one line would cost more than the line is worth, and
/// silently mis-ordering `0.2.0-alpha.1` against `0.2.0` would be worse than
/// declining to answer.
#[cfg(feature = "update-check")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Version {
    major: u64,
    minor: u64,
    patch: u64,
}

#[cfg(feature = "update-check")]
impl Version {
    /// Parses `major.minor.patch`, returning `None` for anything else.
    fn parse(raw: &str) -> Option<Self> {
        if raw.contains(['-', '+']) {
            return None;
        }
        let mut parts = raw.split('.');
        let major = parts.next()?.parse().ok()?;
        let minor = parts.next()?.parse().ok()?;
        let patch = parts.next()?.parse().ok()?;
        if parts.next().is_some() {
            return None;
        }
        Some(Self {
            major,
            minor,
            patch,
        })
    }
}

#[cfg(all(test, feature = "update-check"))]
mod tests {
    use super::*;

    #[test]
    fn index_path_follows_the_registry_protocol() {
        assert_eq!(index_path("a"), "1/a");
        assert_eq!(index_path("ab"), "2/ab");
        assert_eq!(index_path("abc"), "3/a/abc");
        assert_eq!(index_path("incin"), "in/ci/incin");
        assert_eq!(index_path("serde"), "se/rd/serde");
    }

    #[test]
    fn newest_release_skips_yanked_versions() {
        let body = "\
{\"name\":\"incin\",\"vers\":\"0.1.0\",\"yanked\":false}
{\"name\":\"incin\",\"vers\":\"0.3.0\",\"yanked\":true}
{\"name\":\"incin\",\"vers\":\"0.2.0\",\"yanked\":false}
";
        assert_eq!(newest_release(body).as_deref(), Some("0.2.0"));
    }

    #[test]
    fn newest_release_ignores_pre_releases() {
        let body = "\
{\"vers\":\"0.1.0\",\"yanked\":false}
{\"vers\":\"0.2.0-alpha.1\",\"yanked\":false}
";
        assert_eq!(newest_release(body).as_deref(), Some("0.1.0"));
    }

    #[test]
    fn newest_release_tolerates_junk_lines() {
        let body = "not json\n{\"vers\":\"1.2.3\",\"yanked\":false}\n\n";
        assert_eq!(newest_release(body).as_deref(), Some("1.2.3"));
    }

    #[test]
    fn compare_reports_an_available_update() {
        assert_eq!(
            compare("0.1.0", "0.2.0"),
            UpdateStatus::Available {
                running: String::from("0.1.0"),
                latest: String::from("0.2.0"),
            }
        );
    }

    #[test]
    fn compare_treats_a_newer_local_build_as_current() {
        assert_eq!(
            compare("0.9.0", "0.2.0"),
            UpdateStatus::Current {
                running: String::from("0.9.0"),
            }
        );
    }

    #[test]
    fn version_ordering_is_numeric_not_lexicographic() {
        let ten = Version::parse("0.10.0").expect("parses");
        let nine = Version::parse("0.9.0").expect("parses");
        assert!(ten > nine, "0.10.0 must sort above 0.9.0");
    }
}
