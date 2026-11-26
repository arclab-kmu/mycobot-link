//! myCobot 280 serial frames.
//!
//! `FE FE LEN CMD DATA… FA`, with `LEN = len(DATA) + 2`. Multi-byte values are
//! big-endian `i16`; single-byte arguments go in as they are. This mirrors
//! `DataProcessor._mesg` in pymycobot 4.0.7 `common.py` for the 280, which is
//! not in its CRC class list and so ends with `0xFA` instead of a CRC.
//!
//! `tests/golden.rs` holds this module to bytes recorded from pymycobot itself
//! (`tools/golden_frames.py`), not to a re-reading of its source.

pub const HEADER: u8 = 0xFE;
pub const FOOTER: u8 = 0xFA;

/// Command bytes, from pymycobot 4.0.7 `ProtocolCode`.
pub mod code {
    pub const GET_ERROR_INFO: u8 = 0x07;
    pub const IS_POWER_ON: u8 = 0x12;
    pub const SET_FRESH_MODE: u8 = 0x16;
    pub const GET_ANGLES: u8 = 0x20;
    pub const SEND_ANGLES: u8 = 0x22;
    pub const STOP: u8 = 0x29;
    pub const GET_GRIPPER_VALUE: u8 = 0x65;
    pub const SET_GRIPPER_VALUE: u8 = 0x67;
}

/// pymycobot's reader drops a length byte outside this range as noise.
const MAX_LEN: u8 = 30;

fn frame(cmd: u8, data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + 5);
    out.extend_from_slice(&[HEADER, HEADER, data.len() as u8 + 2, cmd]);
    out.extend_from_slice(data);
    out.push(FOOTER);
    out
}

/// pymycobot's `_angle2int`: `int(angle * 100)`.
///
/// `int()` truncates toward zero and so does `as`, on the same IEEE product -
/// which is why 0.29 deg goes out as 28, not 29. Callers pass angles that
/// [`crate::safety`] has already bounded, far inside `i16`.
pub fn angle_to_wire(angle_deg: f64) -> i16 {
    (angle_deg * 100.0) as i16
}

pub fn get_angles() -> Vec<u8> {
    frame(code::GET_ANGLES, &[])
}

pub fn send_angles(angles_deg: &[f64; 6], speed: u8) -> Vec<u8> {
    let mut data = Vec::with_capacity(13);
    for &a in angles_deg {
        data.extend_from_slice(&angle_to_wire(a).to_be_bytes());
    }
    data.push(speed);
    frame(code::SEND_ANGLES, &data)
}

pub fn stop() -> Vec<u8> {
    frame(code::STOP, &[])
}

/// 1 = always run the newest command; 0 = queue them.
pub fn set_fresh_mode(mode: u8) -> Vec<u8> {
    frame(code::SET_FRESH_MODE, &[mode])
}

/// No `gripper_type` byte, as `mycobot-control` calls it.
pub fn set_gripper_value(value: u8, speed: u8) -> Vec<u8> {
    frame(code::SET_GRIPPER_VALUE, &[value, speed])
}

pub fn get_gripper_value() -> Vec<u8> {
    frame(code::GET_GRIPPER_VALUE, &[])
}

pub fn is_power_on() -> Vec<u8> {
    frame(code::IS_POWER_ON, &[])
}

pub fn get_error_information() -> Vec<u8> {
    frame(code::GET_ERROR_INFO, &[])
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    pub cmd: u8,
    pub data: Vec<u8>,
}

/// Incremental frame parser: push whatever the port returned, pull frames.
///
/// Resynchronises on noise the way pymycobot's reader does: a bad length or a
/// missing footer discards one byte and scans for the next `FE FE`.
#[derive(Debug, Default)]
pub struct FrameReader {
    buf: Vec<u8>,
}

impl FrameReader {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    pub fn clear(&mut self) {
        self.buf.clear();
    }

    pub fn next_frame(&mut self) -> Option<Frame> {
        loop {
            let start = self.buf.windows(2).position(|w| w == [HEADER, HEADER]);
            match start {
                Some(i) => {
                    self.buf.drain(..i);
                }
                None => {
                    // Keep a trailing FE: it may be the first half of a header.
                    let keep = usize::from(self.buf.last() == Some(&HEADER));
                    self.buf.drain(..self.buf.len() - keep);
                    return None;
                }
            }
            if self.buf.len() < 3 {
                return None;
            }
            let len = self.buf[2];
            if !(2..=MAX_LEN).contains(&len) {
                self.buf.drain(..1);
                continue;
            }
            let total = usize::from(len) + 3;
            if self.buf.len() < total {
                return None;
            }
            if self.buf[total - 1] != FOOTER {
                self.buf.drain(..1);
                continue;
            }
            let frame = Frame {
                cmd: self.buf[3],
                data: self.buf[4..total - 1].to_vec(),
            };
            self.buf.drain(..total);
            return Some(frame);
        }
    }
}

/// Six joint angles from a `GET_ANGLES` reply, or `None` if it is not one.
///
/// Like `valid_vector()` in `mycobot-control` and the vendor's
/// `listen_real_service.py`, a vector holding exactly -1 deg is treated as a
/// failed read. TODO(unverified): whether the firmware marks a failed joint
/// that way; copying the vendor costs a retry when a joint rests at -1.00.
pub fn decode_angles(frame: &Frame) -> Option<[f64; 6]> {
    if frame.cmd != code::GET_ANGLES || frame.data.len() != 12 {
        return None;
    }
    let mut out = [0.0; 6];
    for (i, pair) in frame.data.as_chunks::<2>().0.iter().enumerate() {
        out[i] = f64::from(i16::from_be_bytes(*pair)) / 100.0;
    }
    if out.contains(&-1.0) {
        return None;
    }
    Some(out)
}

/// The gripper's reported opening, 0 (closed) .. 100 (open). The 280 firmware answers with one
/// byte; a two-byte answer is read as a big-endian i16, as `decode_angles` reads its values.
pub fn decode_gripper(frame: &Frame) -> Option<u8> {
    if frame.cmd != code::GET_GRIPPER_VALUE {
        return None;
    }
    let v = match frame.data.as_slice() {
        [b] => i32::from(*b),
        [hi, lo] => i32::from(i16::from_be_bytes([*hi, *lo])),
        _ => return None,
    };
    u8::try_from(v).ok().filter(|v| *v <= 100)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_is_five_bytes() {
        assert_eq!(stop(), vec![0xFE, 0xFE, 0x02, 0x29, 0xFA]);
    }

    #[test]
    fn angles_truncate_toward_zero_like_python_int() {
        assert_eq!(angle_to_wire(0.29), 28); // 28.999999999999996
        assert_eq!(angle_to_wire(12.34), 1234);
        assert_eq!(angle_to_wire(-0.005), 0);
        assert_eq!(angle_to_wire(-45.678), -4567);
    }

    #[test]
    fn reader_skips_noise_and_split_frames() {
        let mut r = FrameReader::new();
        r.push(&[0x00, 0xFE, 0xFE, 0x63, 0x01, 0xFE]); // bad length, then half a header
        assert_eq!(r.next_frame(), None);
        r.push(&[0xFE, 0x02, 0x29]);
        assert_eq!(r.next_frame(), None);
        r.push(&[0xFA]);
        assert_eq!(
            r.next_frame(),
            Some(Frame {
                cmd: 0x29,
                data: vec![]
            })
        );
        assert_eq!(r.next_frame(), None);
    }

    #[test]
    fn reader_drops_a_frame_without_footer() {
        let mut r = FrameReader::new();
        r.push(&[0xFE, 0xFE, 0x02, 0x29, 0x00]);
        r.push(&stop());
        assert_eq!(
            r.next_frame(),
            Some(Frame {
                cmd: 0x29,
                data: vec![]
            })
        );
    }

    #[test]
    fn gripper_value_decodes_one_or_two_bytes() {
        let f = |data: Vec<u8>| {
            decode_gripper(&Frame {
                cmd: code::GET_GRIPPER_VALUE,
                data,
            })
        };
        assert_eq!(f(vec![37]), Some(37));
        assert_eq!(f(vec![0, 64]), Some(64));
        assert_eq!(f(vec![0, 200]), None);
        assert_eq!(f(vec![]), None);
        assert_eq!(
            decode_gripper(&Frame {
                cmd: code::GET_ANGLES,
                data: vec![5]
            }),
            None
        );
    }

    #[test]
    fn minus_one_degree_reads_as_no_reading() {
        let mut data = Vec::new();
        for v in [0i16, -100, 0, 0, 0, 0] {
            data.extend_from_slice(&v.to_be_bytes());
        }
        assert_eq!(
            decode_angles(&Frame {
                cmd: code::GET_ANGLES,
                data
            }),
            None
        );
    }
}
