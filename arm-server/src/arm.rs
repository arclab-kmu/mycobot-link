//! What the server drives: a dry-run arm by default, the UART only with `--live`.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::time::{Duration, Instant};

use crate::frame::{self, FrameReader};

pub trait Arm {
    /// `Ok(None)` is "no usable reading this time", not an I/O failure.
    fn read_angles(&mut self) -> io::Result<Option<[f64; 6]>>;
    fn send_angles(&mut self, angles_deg: &[f64; 6], speed: u8) -> io::Result<()>;
    fn set_gripper(&mut self, value: u8, speed: u8) -> io::Result<()>;
    fn stop(&mut self) -> io::Result<()>;
    fn is_live(&self) -> bool;
}

impl<A: Arm + ?Sized> Arm for Box<A> {
    fn read_angles(&mut self) -> io::Result<Option<[f64; 6]>> {
        (**self).read_angles()
    }
    fn send_angles(&mut self, angles_deg: &[f64; 6], speed: u8) -> io::Result<()> {
        (**self).send_angles(angles_deg, speed)
    }
    fn set_gripper(&mut self, value: u8, speed: u8) -> io::Result<()> {
        (**self).set_gripper(value, speed)
    }
    fn stop(&mut self) -> io::Result<()> {
        (**self).stop()
    }
    fn is_live(&self) -> bool {
        (**self).is_live()
    }
}

/// Moves nothing. `send_angles` teleports the stored angles, as
/// `mycobot-control`'s `DryRunBackend` does: a dry run proves the link, the
/// gate and the command sequence, not where the arm can go.
#[derive(Debug, Default)]
pub struct DryRunArm {
    pub angles: [f64; 6],
    pub gripper: u8,
    /// Commands in call order, for tests and the exit summary.
    pub commands: Vec<String>,
    /// Lets tests drive the failed-read paths.
    pub fail_reads: bool,
}

impl Arm for DryRunArm {
    fn read_angles(&mut self) -> io::Result<Option<[f64; 6]>> {
        Ok(if self.fail_reads {
            None
        } else {
            Some(self.angles)
        })
    }

    fn send_angles(&mut self, angles_deg: &[f64; 6], speed: u8) -> io::Result<()> {
        self.commands
            .push(format!("send_angles {angles_deg:?} {speed}"));
        self.angles = *angles_deg;
        Ok(())
    }

    fn set_gripper(&mut self, value: u8, speed: u8) -> io::Result<()> {
        self.commands
            .push(format!("set_gripper_value {value} {speed}"));
        self.gripper = value;
        Ok(())
    }

    fn stop(&mut self) -> io::Result<()> {
        self.commands.push("stop".into());
        Ok(())
    }

    fn is_live(&self) -> bool {
        false
    }
}

/// The vendor ROS 2 nodes and `mycobot-control` serialise UART access with an
/// flock on this file. Holding it for the whole session makes them refuse each
/// other instead of interleaving frames.
pub const VENDOR_LOCK_PATH: &str = "/tmp/mycobot_lock";

/// The Jetson's UART to the Atom board.
pub struct SerialArm {
    port: Box<dyn serialport::SerialPort>,
    reader: FrameReader,
    reply_timeout: Duration,
    _lock: File,
}

impl SerialArm {
    /// Take the vendor lock, open the port, and set fresh mode 1 (newest
    /// command wins), as `mycobot-control` does by default.
    pub fn open(path: &str, baud: u32, reply_timeout: Duration) -> io::Result<Self> {
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(VENDOR_LOCK_PATH)?;
        lock.try_lock().map_err(|_| {
            io::Error::new(
                io::ErrorKind::WouldBlock,
                format!(
                    "{VENDOR_LOCK_PATH} is held by another process - a vendor ROS 2 node or a mycobot-* \
                     command. Only one process may own the arm's serial port."
                ),
            )
        })?;
        let port = serialport::new(path, baud)
            .timeout(Duration::from_millis(5))
            .open()
            .map_err(io::Error::other)?;
        let mut arm = SerialArm {
            port,
            reader: FrameReader::new(),
            reply_timeout,
            _lock: lock,
        };
        arm.write(&frame::set_fresh_mode(1))?;
        Ok(arm)
    }

    fn write(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.port.write_all(bytes)?;
        self.port.flush()
    }

    /// Send a query and wait for a frame with the same command byte and some
    /// data. That skips an echo of the request itself, as pymycobot's reader
    /// does; replies to earlier fire-and-forget commands go with the input
    /// buffer first.
    fn query(&mut self, request: &[u8]) -> io::Result<Option<frame::Frame>> {
        self.port
            .clear(serialport::ClearBuffer::Input)
            .map_err(io::Error::other)?;
        self.reader.clear();
        self.write(request)?;
        let deadline = Instant::now() + self.reply_timeout;
        let mut buf = [0u8; 64];
        while Instant::now() < deadline {
            match self.port.read(&mut buf) {
                Ok(n) => self.reader.push(&buf[..n]),
                Err(e) if e.kind() == io::ErrorKind::TimedOut => {}
                Err(e) => return Err(e),
            }
            while let Some(f) = self.reader.next_frame() {
                if f.cmd == request[3] && !f.data.is_empty() {
                    return Ok(Some(f));
                }
            }
        }
        Ok(None)
    }
}

/// Last line of defence: however the process ends (a panic unwinding
/// included), the arm gets one more stop frame. After a clean shutdown that
/// is a second, harmless stop.
impl Drop for SerialArm {
    fn drop(&mut self) {
        let _ = self.write(&frame::stop());
    }
}

impl Arm for SerialArm {
    fn read_angles(&mut self) -> io::Result<Option<[f64; 6]>> {
        Ok(self
            .query(&frame::get_angles())?
            .as_ref()
            .and_then(frame::decode_angles))
    }

    /// Fire and forget. pymycobot 4.0.7 waits 0.5 s for a reply to every
    /// synchronous call, this one and stop() included, and writes the frame 3
    /// times when none comes (recorded over a pty by tools/golden_frames.py).
    /// Whether the 280 firmware replies at all is TODO(unverified); a re-send
    /// is a second command, so send once.
    fn send_angles(&mut self, angles_deg: &[f64; 6], speed: u8) -> io::Result<()> {
        self.write(&frame::send_angles(angles_deg, speed))
    }

    fn set_gripper(&mut self, value: u8, speed: u8) -> io::Result<()> {
        self.write(&frame::set_gripper_value(value, speed))
    }

    fn stop(&mut self) -> io::Result<()> {
        self.write(&frame::stop())
    }

    fn is_live(&self) -> bool {
        true
    }
}
