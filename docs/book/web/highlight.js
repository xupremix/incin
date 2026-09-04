/* Rust syntax highlighting, shared by the book and the capability reference.
   It lived inside book.js, which the reference page does not load; copying it
   would have left two highlighters to keep in step, so it is one file that
   both pages use. The `hl-*` classes it emits are styled in book.css, which
   both pages already load. */
(function (global) {
  "use strict";

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function highlightRustCode(code) {
    const tokens = [];
    let text = code;

    // Pattern definitions
    const RUST_RE = new RegExp(
      [
        '(?<comment>//[^\n]*|/\\*[\\s\\S]*?\\*/)',
        '(?<attribute>#!?\\[[\\s\\S]*?\\])',
        '(?<string>"(?:\\\\.|[^"\\\\])*"|b"(?:\\\\.|[^"\\\\])*"|r#*"(?:[\\s\\S]*?)"#*)',
        '(?<char>\'(?:\\\\.|[^\'\\\\])\'|b\'(?:\\\\.|[^\'\\\\])\')',
        '(?<lifetime>\'[a-zA-Z_]\\w*\\b)',
        '(?<keyword>\\b(?:as|async|await|break|const|continue|crate|dyn|else|enum|extern|false|fn|for|if|impl|in|let|loop|match|mod|move|mut|pub|ref|return|self|Self|static|struct|super|trait|true|type|unsafe|use|where|while)\\b)',
        '(?<type>\\b(?:Tensor|Backend|Shape|Dim|DimCons|Nil|DType|Device|ConstDevice|ConstDType|Cpu|Cuda|DefaultBackend|Grad|NoGrad|Result|Option|Some|None|Ok|Err|String|Vec|Box|Arc|Rc|PhantomData|Unsigned|UInt|UTerm|Dyn|Ranked|f32|f64|i8|i16|i32|i64|i128|isize|u8|u16|u32|u64|u128|usize|bool|char|str)\\b)',
        '(?<macro>\\b[a-zA-Z_]\\w*!)',
        '(?<number>\\b(?:0x[0-9a-fA-F_]+|0b[01_]+|0o[0-7_]+|\\d[\\d_]*(?:\\.[\\d_]+)?(?:[eE][+-]?[\\d_]+)?(?:f32|f64|i8|i16|i32|i64|isize|u8|u16|u32|u64|usize)?)\\b)',
        '(?<fn>\\b[a-zA-Z_]\\w*(?=\\s*\\())',
      ].join('|'),
      'g'
    );

    let lastIndex = 0;
    let result = '';
    let match;

    while ((match = RUST_RE.exec(text)) !== null) {
      result += escapeHtml(text.slice(lastIndex, match.index));
      const groups = match.groups;
      if (groups.comment) {
        result += '<span class="hl-cmt">' + escapeHtml(match[0]) + '</span>';
      } else if (groups.attribute) {
        result += '<span class="hl-meta">' + escapeHtml(match[0]) + '</span>';
      } else if (groups.string || groups.char) {
        result += '<span class="hl-str">' + escapeHtml(match[0]) + '</span>';
      } else if (groups.lifetime) {
        result += '<span class="hl-sym">' + escapeHtml(match[0]) + '</span>';
      } else if (groups.keyword) {
        result += '<span class="hl-kw">' + escapeHtml(match[0]) + '</span>';
      } else if (groups.type) {
        result += '<span class="hl-type">' + escapeHtml(match[0]) + '</span>';
      } else if (groups.macro) {
        result += '<span class="hl-macro">' + escapeHtml(match[0]) + '</span>';
      } else if (groups.number) {
        result += '<span class="hl-num">' + escapeHtml(match[0]) + '</span>';
      } else if (groups.fn) {
        result += '<span class="hl-fn">' + escapeHtml(match[0]) + '</span>';
      } else {
        result += escapeHtml(match[0]);
      }
      lastIndex = RUST_RE.lastIndex;
    }
    result += escapeHtml(text.slice(lastIndex));
    return result;
  }


  global.incinHighlightRust = highlightRustCode;
  global.incinEscapeHtml = escapeHtml;
}(window));
