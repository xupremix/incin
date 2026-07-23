//! Runtime configuration, read once from the environment at startup — this
//! is a developer-machine proxy launched by an editor, not a deployed
//! service, so a config file would be more machinery than the surface needs.

/// Environment variable naming the real rust-analyzer binary to spawn.
/// Falls back to `"rust-analyzer"`, resolved via `PATH`.
pub const RA_PATH_VAR: &str = "KINDLE_LSP_RA_PATH";
/// Set to `"0"` to disable inlay-hint label rewriting (diagnostics are
/// still humanized regardless of this flag).
pub const HINTS_VAR: &str = "KINDLE_LSP_HINTS";
/// Set to `"1"` to also drop the backend/dtype/grad tail from a rewritten
/// inlay hint (`Tensor<[2, 3]>` instead of `Tensor<[2, 3], CpuBackendImpl<f32, Cpu>>`).
pub const SHORTEN_BACKEND_VAR: &str = "KINDLE_LSP_SHORTEN_BACKEND";

/// The proxy's resolved configuration for one run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    /// Path (or bare name, resolved via `PATH`) of the rust-analyzer binary to spawn.
    pub ra_path: String,
    /// Whether inlay-hint labels get rewritten at all.
    pub hints_enabled: bool,
    /// Whether a rewritten hint also drops the backend/dtype/grad tail.
    pub shorten_backend: bool,
}

impl Config {
    /// Reads the configuration from the process environment.
    pub fn from_env() -> Self {
        Self {
            ra_path: std::env::var(RA_PATH_VAR).unwrap_or_else(|_| "rust-analyzer".to_string()),
            hints_enabled: std::env::var(HINTS_VAR).map(|v| v != "0").unwrap_or(true),
            shorten_backend: std::env::var(SHORTEN_BACKEND_VAR)
                .map(|v| v == "1")
                .unwrap_or(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Both cases live in one test function (rather than two `#[test]`s) so
    // they can't run concurrently on separate threads and race on these
    // process-global environment variables.
    #[test]
    fn from_env_defaults_and_overrides() {
        // SAFETY: no other test in this crate reads/writes these env vars.
        unsafe {
            std::env::remove_var(RA_PATH_VAR);
            std::env::remove_var(HINTS_VAR);
            std::env::remove_var(SHORTEN_BACKEND_VAR);
        }
        let config = Config::from_env();
        assert_eq!(config.ra_path, "rust-analyzer");
        assert!(config.hints_enabled);
        assert!(!config.shorten_backend);

        // SAFETY: no other test in this crate reads/writes these env vars.
        unsafe {
            std::env::set_var(RA_PATH_VAR, "/opt/ra/rust-analyzer");
            std::env::set_var(HINTS_VAR, "0");
            std::env::set_var(SHORTEN_BACKEND_VAR, "1");
        }
        let config = Config::from_env();
        assert_eq!(config.ra_path, "/opt/ra/rust-analyzer");
        assert!(!config.hints_enabled);
        assert!(config.shorten_backend);

        // SAFETY: no other test in this crate reads/writes these env vars.
        unsafe {
            std::env::remove_var(RA_PATH_VAR);
            std::env::remove_var(HINTS_VAR);
            std::env::remove_var(SHORTEN_BACKEND_VAR);
        }
    }
}
