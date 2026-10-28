//! myCobot 280 JN arm server.
//!
//! The brain (Python, DGX Spark) proposes; this process disposes. Every check
//! that keeps the arm safe lives here, on the Jetson, so a stalled brain or a
//! dropped link ends in `stop()` rather than in a last command replayed.
//!
//! - [`frame`]: the `FE FE … FA` serial frames, byte-for-byte with pymycobot 4.0.7.
//! - [`protocol`]: the JSON-lines messages in `schema/messages.schema.json`.
//! - [`safety`]: limits ported from `mycobot-control`'s `safety/limits.py`.
//! - [`arm`]: the [`arm::Arm`] trait, a dry-run arm and the serial arm.
//! - [`server`]: gate, watchdog and request handling, with time passed in.
//! - [`lines`]: newline framing for the TCP stream.
//! - [`log`]: stderr lines that can never take the process down.

pub mod arm;
pub mod frame;
pub mod lines;
pub mod protocol;
pub mod safety;
pub mod server;

/// One `[mycobot-arm]` line on stderr; a write error is dropped.
///
/// Not `eprintln!`: it panics when stderr is a closed pipe, and with
/// `mycobot-arm … | tee log` a Ctrl-C kills `tee` first. The panic then came
/// from the "session ended" line, before `on_disconnect` could send `stop()`.
pub fn log(msg: impl AsRef<str>) {
    use std::io::Write;
    let _ = writeln!(std::io::stderr().lock(), "[mycobot-arm] {}", msg.as_ref());
}
