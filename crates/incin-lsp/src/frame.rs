//! Raw LSP `Content-Length`-framed message I/O, independent of any JSON-RPC
//! type modeling - the proxy only needs to locate each message's byte span,
//! not parse its structure, to forward it.

use std::io::{self, BufRead, Write};

/// Reads one `Content-Length`-framed LSP message from `r` and returns its
/// raw JSON body bytes (unparsed). Returns `Ok(None)` on a clean EOF before
/// any header bytes are read (the normal way an LSP stream ends).
pub fn read_frame<R: BufRead>(r: &mut R) -> io::Result<Option<Vec<u8>>> {
    let mut content_length: Option<usize> = None;
    loop {
        let mut header_line = String::new();
        let bytes_read = r.read_line(&mut header_line)?;
        if bytes_read == 0 {
            return Ok(None);
        }
        let line = header_line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            break; // blank line ends the header block
        }
        if let Some(value) = line.strip_prefix("Content-Length:") {
            let parsed = value.trim().parse::<usize>().map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("bad Content-Length: {e}"),
                )
            })?;
            content_length = Some(parsed);
        }
        // Other headers (e.g. Content-Type) are valid per the LSP spec but
        // unused here - ignored, not an error.
    }
    let content_length = content_length.ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "missing Content-Length header")
    })?;
    let mut body = vec![0u8; content_length];
    r.read_exact(&mut body)?;
    Ok(Some(body))
}

/// Writes `body` as one `Content-Length`-framed LSP message to `w`. The
/// header is computed from `body.len()` - the actual byte length - not a
/// char count, so rewritten multi-byte UTF-8 content never desyncs the
/// reader on the other end.
pub fn write_frame<W: Write>(w: &mut W, body: &[u8]) -> io::Result<()> {
    write!(w, "Content-Length: {}\r\n\r\n", body.len())?;
    w.write_all(body)?;
    w.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufReader, Cursor};

    #[test]
    fn round_trips_a_simple_ascii_message() {
        let mut buf = Vec::new();
        write_frame(&mut buf, b"{\"hello\":true}").unwrap();
        let mut reader = BufReader::new(Cursor::new(buf));
        let body = read_frame(&mut reader).unwrap().unwrap();
        assert_eq!(body, b"{\"hello\":true}");
        assert!(read_frame(&mut reader).unwrap().is_none());
    }

    #[test]
    fn ignores_unrelated_headers_before_content_length() {
        let raw = b"Content-Type: application/vscode-jsonrpc; charset=utf-8\r\nContent-Length: 2\r\n\r\n{}";
        let mut reader = BufReader::new(Cursor::new(raw.to_vec()));
        let body = read_frame(&mut reader).unwrap().unwrap();
        assert_eq!(body, b"{}");
    }

    #[test]
    fn returns_none_on_clean_eof() {
        let mut reader = BufReader::new(Cursor::new(Vec::<u8>::new()));
        assert!(read_frame(&mut reader).unwrap().is_none());
    }

    /// `Content-Length` is a byte count, not a char count - a rewritten
    /// diagnostic containing multi-byte UTF-8 (e.g. the 💡/└── glyphs this
    /// tooling already emits elsewhere) must produce a header matching the
    /// serialized body's actual byte length, or the reader on the other end
    /// desyncs on the very next frame.
    #[test]
    fn content_length_header_is_a_byte_count_not_a_char_count() {
        let body = "{\"message\":\"💡 not ASCII — 3 bytes become more bytes\"}"
            .as_bytes()
            .to_vec();
        assert!(body.len() > body.iter().filter(|&&b| b < 0x80).count());

        let mut buf = Vec::new();
        write_frame(&mut buf, &body).unwrap();

        let header_end = buf.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
        let header = std::str::from_utf8(&buf[..header_end]).unwrap();
        let declared_len: usize = header
            .strip_prefix("Content-Length: ")
            .unwrap()
            .parse()
            .unwrap();
        assert_eq!(declared_len, body.len());

        let mut reader = BufReader::new(Cursor::new(buf));
        let round_tripped = read_frame(&mut reader).unwrap().unwrap();
        assert_eq!(round_tripped, body);
    }

    #[test]
    fn missing_content_length_is_an_error_not_a_panic() {
        let raw = b"Content-Type: text/plain\r\n\r\n{}";
        let mut reader = BufReader::new(Cursor::new(raw.to_vec()));
        assert!(read_frame(&mut reader).is_err());
    }
}
