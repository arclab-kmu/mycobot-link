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

pub mod arm;
pub mod frame;
pub mod lines;
pub mod protocol;
pub mod safety;
pub mod server;
