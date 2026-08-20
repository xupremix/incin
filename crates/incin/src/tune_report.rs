//! `cargo incin tune` - autotune cache management report (`UX-006`).

/// Process exit code for a successful tune command.
pub const EXIT_OK: i32 = 0;
/// Process exit code for invalid tune-command arguments or refused requests.
pub const EXIT_USAGE: i32 = 2;

/// Runs `cargo incin tune` CLI subcommand.
pub fn run(args: &[String]) -> (String, i32) {
    let mut json_mode = false;
    let mut clear_mode = false;
    let mut offline_mode = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--json" => json_mode = true,
            "--clear" => clear_mode = true,
            "--offline" => offline_mode = true,
            "--help" | "-h" => {
                return (
                    "cargo incin tune — inspect and manage autotune cache\n\nUSAGE:\n    cargo incin tune [--json] [--clear] [--offline]\n".to_string(),
                    EXIT_OK,
                );
            }
            _ => {}
        }
        i += 1;
    }

    let cache_dir = std::env::var_os("INCIN_AUTOTUNE_CACHE_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(".cache")
                .join("incin")
                .join("autotune")
        });

    if clear_mode {
        if cache_dir.exists() {
            let path_str = cache_dir.to_string_lossy();
            if path_str == "/"
                || path_str == "."
                || path_str == ".."
                || path_str.is_empty()
                || cache_dir.parent().is_none()
            {
                return (
                    "Error: Refusing to clear root or top-level system directory.\n".to_string(),
                    EXIT_USAGE,
                );
            }
            let _ = std::fs::remove_dir_all(&cache_dir);
        }
        let msg = if json_mode {
            serde_json::json!({
                "status": "cleared",
                "cache_dir": cache_dir.display().to_string()
            })
            .to_string()
                + "\n"
        } else {
            format!("Autotune cache cleared at {}\n", cache_dir.display())
        };
        return (msg, EXIT_OK);
    }

    let exists = cache_dir.exists();
    let cache_path_str = cache_dir.display().to_string();

    if json_mode {
        let json = serde_json::json!({
            "cache_dir": cache_path_str,
            "exists": exists,
            "offline": offline_mode,
            "status": if exists { "active" } else { "empty" },
        });
        (
            serde_json::to_string_pretty(&json).unwrap_or_default() + "\n",
            EXIT_OK,
        )
    } else {
        let mut out = String::new();
        out.push_str("Autotune Cache Report:\n");
        out.push_str(&format!("  • Cache Directory: {}\n", cache_path_str));
        out.push_str(&format!(
            "  • State: {}\n",
            if exists { "active" } else { "empty" }
        ));
        out.push_str(&format!(
            "  • Mode: {}\n",
            if offline_mode { "offline" } else { "online" }
        ));
        (out, EXIT_OK)
    }
}
