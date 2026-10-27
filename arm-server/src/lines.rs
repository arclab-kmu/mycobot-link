//! Newline framing for the TCP stream.
//!
//! Hand-rolled rather than `BufRead::read_line`: the socket has a short read
//! timeout so the loop can keep ticking, and a timeout must never cost the
//! half line already received.

pub const MAX_LINE: usize = 64 * 1024;

#[derive(Debug, Default)]
pub struct LineBuffer {
    buf: Vec<u8>,
}

impl LineBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Complete lines in arrival order, without the `\n` (or `\r\n`). Blank
    /// lines are skipped. An over-long line is an error: the peer is not
    /// speaking this protocol and the connection should drop.
    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<String>, String> {
        self.buf.extend_from_slice(bytes);
        let mut lines = Vec::new();
        while let Some(i) = self.buf.iter().position(|&b| b == b'\n') {
            let raw: Vec<u8> = self.buf.drain(..=i).collect();
            let text = String::from_utf8_lossy(&raw[..i]);
            let text = text.trim_end_matches('\r');
            if !text.trim().is_empty() {
                lines.push(text.to_owned());
            }
        }
        if self.buf.len() > MAX_LINE {
            return Err(format!("line longer than {MAX_LINE} bytes"));
        }
        Ok(lines)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_and_joined_lines() {
        let mut lb = LineBuffer::new();
        assert_eq!(lb.push(b"{\"a\"").unwrap(), Vec::<String>::new());
        assert_eq!(
            lb.push(b":1}\n\n{\"b\":2}\r\n{").unwrap(),
            vec!["{\"a\":1}", "{\"b\":2}"]
        );
        assert_eq!(lb.push(b"}\n").unwrap(), vec!["{}"]);
    }

    #[test]
    fn overlong_line_is_an_error() {
        let mut lb = LineBuffer::new();
        assert!(lb.push(&vec![b'x'; MAX_LINE + 1]).is_err());
    }
}
