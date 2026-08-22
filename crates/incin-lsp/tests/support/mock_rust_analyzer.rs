//! A std-only rust-analyzer stand-in compiled by `proxy_integration.rs`.
//!
//! It deliberately lives outside `src/bin`: it is test support, not an
//! installable product binary. The fixture reads one inlay-hint request and
//! returns canned frames that exercise the proxy's byte-preserving and
//! rewriting paths.

use std::io::{self, BufRead, BufReader, Write};

fn read_frame(reader: &mut impl BufRead) -> io::Result<Option<Vec<u8>>> {
    let mut content_length = None;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            return Ok(None);
        }
        if line == "\r\n" || line == "\n" {
            break;
        }
        if let Some(value) = line.strip_prefix("Content-Length:") {
            content_length = Some(value.trim().parse::<usize>().map_err(|error| {
                io::Error::new(io::ErrorKind::InvalidData, format!("invalid length: {error}"))
            })?);
        }
    }

    let length = content_length.ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "missing Content-Length header")
    })?;
    let mut body = vec![0; length];
    reader.read_exact(&mut body)?;
    Ok(Some(body))
}

fn write_frame(writer: &mut impl Write, body: &[u8]) -> io::Result<()> {
    write!(writer, "Content-Length: {}\r\n\r\n", body.len())?;
    writer.write_all(body)
}

fn request_id(body: &[u8]) -> &[u8] {
    let Some(index) = body.windows(4).position(|window| window == b"\"id\"") else {
        return b"null";
    };
    let Some(colon) = body[index + 4..].iter().position(|byte| *byte == b':') else {
        return b"null";
    };
    let value = &body[index + 4 + colon + 1..];
    let start = value.iter().position(|byte| !byte.is_ascii_whitespace()).unwrap_or(0);
    let end = value[start..]
        .iter()
        .position(|byte| matches!(*byte, b',' | b'}'))
        .map(|offset| start + offset)
        .unwrap_or(value.len());
    &value[start..end]
}

fn main() -> io::Result<()> {
    if std::env::args().skip(1).any(|argument| argument == "--version") {
        println!("mock-rust-analyzer 0.1.0");
        return Ok(());
    }

    let mut stdin = BufReader::new(io::stdin());
    let request = read_frame(&mut stdin)?.expect("harness must send one frame");
    let id = request_id(&request);
    let mut stdout = io::stdout();

    let method_is_resolve = request.windows(17).any(|w| w == b"inlayHint/resolve");
    if method_is_resolve {
        write_frame(
            &mut stdout,
            format!(r#"{{"jsonrpc":"2.0","id":{},"result":{{"position":{{"line":3,"character":12}},"label":"Tensor<(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B1>), CpuBackendImpl<Cpu>>"}}}}"#, String::from_utf8_lossy(id)).as_bytes(),
        )?;
        return Ok(());
    }

    write_frame(
        &mut stdout,
        br#"{"jsonrpc":"2.0","method":"window/logMessage","params":{"type":3,"message":"mock server started"}}"#,
    )?;
    write_frame(
        &mut stdout,
        br#"{"jsonrpc":"2.0","method":"textDocument/publishDiagnostics","params":{"uri":"file:///model.rs","diagnostics":[{"range":{"start":{"line":10,"character":14},"end":{"line":10,"character":21}},"severity":1,"message":"Cannot reshape: source has UInt<UInt<UInt<UTerm, B1>, B1>, B0> elements but the target shape has UInt<UInt<UInt<UInt<UTerm, B1>, B0>, B0>, B0> elements"}]}}"#,
    )?;
    write_frame(
        &mut stdout,
        format!(r#"{{"jsonrpc":"2.0","id":{},"result":[{{"position":{{"line":3,"character":12}},"label":"Tensor<(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B1>), CpuBackendImpl<Cpu>>"}}]}}"#, String::from_utf8_lossy(id)).as_bytes(),
    )?;
    Ok(())
}
