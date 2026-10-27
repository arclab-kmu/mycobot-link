//! Gate, watchdog and request handling.
//!
//! No I/O of its own and no clock of its own: the caller passes `now`, so the
//! watchdog and freshness rules are tested without sleeping. `main.rs` owns
//! the socket and calls [`Server::handle_line`] and [`Server::tick`].

use std::time::{Duration, Instant};

use crate::arm::Arm;
use crate::protocol::{Event, PROTOCOL_VERSION, Request, parse_request};
use crate::safety;

#[derive(Debug, Clone)]
pub struct Config {
    /// Gate open and no line from the brain for this long: stop.
    pub watchdog: Duration,
    pub state_period: Duration,
    /// Oldest joint reading a command may be checked against.
    pub max_read_age: Duration,
    pub max_speed: u32,
    pub max_delta_deg: f64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            watchdog: Duration::from_millis(500),
            state_period: Duration::from_millis(100),
            max_read_age: Duration::from_millis(300),
            max_speed: safety::FAST_SPEED,
            max_delta_deg: safety::MAX_COMMAND_DELTA_DEG,
        }
    }
}

pub struct Server<A: Arm> {
    arm: A,
    cfg: Config,
    gate_open: bool,
    gate_reason: String,
    last_rx: Option<Instant>,
    last_reading: Option<([f64; 6], Instant)>,
    last_read_ok: bool,
    next_state_at: Option<Instant>,
    seq: u64,
}

impl<A: Arm> Server<A> {
    pub fn new(arm: A, cfg: Config) -> Self {
        Server {
            arm,
            cfg,
            gate_open: false,
            gate_reason: "never opened".into(),
            last_rx: None,
            last_reading: None,
            last_read_ok: false,
            next_state_at: None,
            seq: 0,
        }
    }

    pub fn arm(&self) -> &A {
        &self.arm
    }

    pub fn gate_open(&self) -> bool {
        self.gate_open
    }

    pub fn gate_reason(&self) -> &str {
        &self.gate_reason
    }

    /// A new brain connection: gate closed until it says `enable`, hello
    /// first, then a state on the very next tick.
    pub fn on_connect(&mut self, now: Instant) -> Vec<Event> {
        self.close_gate("new connection");
        self.last_rx = Some(now);
        self.next_state_at = Some(now);
        vec![Event::Hello {
            protocol: PROTOCOL_VERSION,
            server: concat!("mycobot-arm/", env!("CARGO_PKG_VERSION")).into(),
            live: self.arm.is_live(),
            watchdog_ms: self.cfg.watchdog.as_millis() as u64,
            state_hz: 1.0 / self.cfg.state_period.as_secs_f64(),
        }]
    }

    /// Link lost: stop whatever the gate says. A move sent just before a
    /// `disable` may still be running, and nobody is watching it now.
    pub fn on_disconnect(&mut self, why: &str) {
        self.halt(&format!("link lost: {why}"));
        self.last_rx = None;
        self.next_state_at = None;
    }

    pub fn shutdown(&mut self) {
        self.halt("server shutting down");
    }

    pub fn handle_line(&mut self, line: &str, now: Instant) -> Vec<Event> {
        self.last_rx = Some(now);
        match parse_request(line) {
            Err((id, why)) => vec![Event::rejected(id, why)],
            Ok(Request::Ping {}) => vec![],
            Ok(Request::Enable { id }) => vec![self.enable(id, now)],
            Ok(Request::Disable { id }) => {
                self.close_gate("disabled by brain");
                vec![Event::ok(id)]
            }
            Ok(Request::Stop { id }) => {
                // Accepted even with the gate closed: stop must always get through.
                match self.arm.stop() {
                    Ok(()) => {
                        self.close_gate("stopped by brain");
                        vec![Event::ok(id)]
                    }
                    Err(e) => {
                        self.close_gate(&format!("stop() failed: {e}"));
                        vec![Event::rejected(Some(id), format!("stop() failed: {e}"))]
                    }
                }
            }
            Ok(Request::MoveJoints {
                id,
                angles_deg,
                speed,
            }) => vec![self.move_joints(id, &angles_deg, speed, now)],
            Ok(Request::Gripper { id, value, speed }) => vec![self.gripper(id, value, speed)],
        }
    }

    /// Watchdog and freshness first, then a state message when one is due.
    pub fn tick(&mut self, now: Instant) -> Vec<Event> {
        if self.gate_open
            && let Some(rx) = self.last_rx
        {
            let quiet = now.saturating_duration_since(rx);
            if quiet > self.cfg.watchdog {
                self.halt(&format!(
                    "watchdog: no line from the brain for {} ms",
                    quiet.as_millis()
                ));
            }
        }
        let Some(due) = self.next_state_at else {
            return vec![];
        };
        if now < due {
            return vec![];
        }
        self.read(now);
        if self.gate_open && self.fresh_reading(now).is_none() {
            self.halt("joint reading stale while the gate was open");
        }
        // Skip missed periods rather than bursting to catch up.
        let mut next = due + self.cfg.state_period;
        if next <= now {
            next = now + self.cfg.state_period;
        }
        self.next_state_at = Some(next);
        self.seq += 1;
        vec![Event::State {
            seq: self.seq,
            angles_deg: if self.last_read_ok {
                self.last_reading.map(|(q, _)| q)
            } else {
                None
            },
            read_age_ms: self
                .last_reading
                .map(|(_, t)| now.saturating_duration_since(t).as_millis() as u64),
            gate_open: self.gate_open,
            gate_reason: self.gate_reason.clone(),
            live: self.arm.is_live(),
        }]
    }

    fn read(&mut self, now: Instant) {
        match self.arm.read_angles() {
            Ok(Some(q)) => {
                self.last_reading = Some((q, now));
                self.last_read_ok = true;
            }
            Ok(None) => self.last_read_ok = false,
            Err(e) => {
                self.last_read_ok = false;
                eprintln!("[mycobot-arm] read_angles failed: {e}");
            }
        }
    }

    fn fresh_reading(&self, now: Instant) -> Option<[f64; 6]> {
        self.last_reading
            .filter(|(_, t)| now.saturating_duration_since(*t) <= self.cfg.max_read_age)
            .map(|(q, _)| q)
    }

    fn stale_reason(&self, now: Instant) -> String {
        match self.last_reading {
            None => "no joint reading yet".into(),
            Some((_, t)) => format!(
                "joint reading is {} ms old (max {})",
                now.saturating_duration_since(t).as_millis(),
                self.cfg.max_read_age.as_millis()
            ),
        }
    }

    fn enable(&mut self, id: u64, now: Instant) -> Event {
        if self.fresh_reading(now).is_none() {
            return Event::rejected(
                Some(id),
                format!("cannot enable: {}", self.stale_reason(now)),
            );
        }
        if !self.gate_open {
            self.gate_open = true;
            self.gate_reason = "enabled by brain".into();
            eprintln!("[mycobot-arm] gate OPEN (enabled by brain)");
        }
        Event::ok(id)
    }

    fn move_joints(&mut self, id: u64, angles_deg: &[f64], speed: u32, now: Instant) -> Event {
        let checked = (|| {
            if !self.gate_open {
                return Err(format!("gate closed: {}", self.gate_reason));
            }
            let target = safety::check_joint_target(angles_deg)?;
            let measured = self
                .fresh_reading(now)
                .ok_or_else(|| self.stale_reason(now))?;
            safety::check_command_delta(&target, &measured, self.cfg.max_delta_deg)?;
            let speed = safety::check_speed(speed, self.cfg.max_speed)?;
            Ok((target, speed))
        })();
        match checked {
            Err(why) => Event::rejected(Some(id), why),
            Ok((target, speed)) => match self.arm.send_angles(&target, speed) {
                Ok(()) => Event::ok(id),
                Err(e) => {
                    self.close_gate(&format!("send_angles failed: {e}"));
                    Event::rejected(Some(id), format!("send_angles failed: {e}"))
                }
            },
        }
    }

    fn gripper(&mut self, id: u64, value: u32, speed: u32) -> Event {
        let checked = (|| {
            if !self.gate_open {
                return Err(format!("gate closed: {}", self.gate_reason));
            }
            Ok((
                safety::check_gripper(value)?,
                safety::check_speed(speed, safety::SPEED_MAX)?,
            ))
        })();
        match checked {
            Err(why) => Event::rejected(Some(id), why),
            Ok((value, speed)) => match self.arm.set_gripper(value, speed) {
                Ok(()) => Event::ok(id),
                Err(e) => {
                    self.close_gate(&format!("set_gripper_value failed: {e}"));
                    Event::rejected(Some(id), format!("set_gripper_value failed: {e}"))
                }
            },
        }
    }

    /// stop() and close the gate; a failed stop() is reported, never retried
    /// in a loop that could starve the rest of the server.
    fn halt(&mut self, why: &str) {
        if let Err(e) = self.arm.stop() {
            eprintln!("[mycobot-arm] stop() failed: {e}");
        }
        self.close_gate(why);
    }

    fn close_gate(&mut self, why: &str) {
        if self.gate_open {
            eprintln!("[mycobot-arm] gate CLOSED ({why})");
        }
        self.gate_open = false;
        self.gate_reason = why.into();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arm::DryRunArm;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    /// Connected, one state read, gate open.
    fn enabled(t0: Instant) -> Server<DryRunArm> {
        let mut s = Server::new(DryRunArm::default(), Config::default());
        s.on_connect(t0);
        assert_eq!(s.tick(t0).len(), 1);
        assert_eq!(
            s.handle_line(r#"{"type":"enable","id":1}"#, t0),
            vec![Event::ok(1)]
        );
        s
    }

    fn reply_ok(events: &[Event]) -> bool {
        matches!(events, [Event::Reply { ok: true, .. }])
    }

    #[test]
    fn gate_starts_closed_and_moves_are_refused() {
        let t0 = Instant::now();
        let mut s = Server::new(DryRunArm::default(), Config::default());
        s.on_connect(t0);
        s.tick(t0);
        let r = s.handle_line(
            r#"{"type":"move_joints","id":2,"angles_deg":[5,0,0,0,0,0],"speed":20}"#,
            t0,
        );
        assert!(
            matches!(&r[..], [Event::Reply { ok: false, error: Some(e), .. }] if e.starts_with("gate closed"))
        );
        assert!(s.arm().commands.is_empty());
    }

    #[test]
    fn enable_needs_a_fresh_reading() {
        let t0 = Instant::now();
        let mut s = Server::new(DryRunArm::default(), Config::default());
        s.on_connect(t0);
        assert!(!reply_ok(&s.handle_line(r#"{"type":"enable","id":1}"#, t0)));
        s.tick(t0);
        assert!(!reply_ok(
            &s.handle_line(r#"{"type":"enable","id":2}"#, t0 + ms(301))
        ));
        assert!(reply_ok(
            &s.handle_line(r#"{"type":"enable","id":3}"#, t0 + ms(300))
        ));
    }

    #[test]
    fn small_move_goes_out_and_a_jump_is_refused() {
        let t0 = Instant::now();
        let mut s = enabled(t0);
        assert!(reply_ok(&s.handle_line(
            r#"{"type":"move_joints","id":2,"angles_deg":[10,0,0,0,0,0],"speed":20}"#,
            t0 + ms(10)
        )));
        // Checked against the measured angles (still 0 until the next read), not the last target.
        assert!(!reply_ok(&s.handle_line(
            r#"{"type":"move_joints","id":3,"angles_deg":[25,0,0,0,0,0],"speed":20}"#,
            t0 + ms(20)
        )));
        assert_eq!(
            s.arm().commands,
            vec!["send_angles [10.0, 0.0, 0.0, 0.0, 0.0, 0.0] 20"]
        );
    }

    #[test]
    fn speed_over_the_cap_is_refused() {
        let t0 = Instant::now();
        let mut s = enabled(t0);
        assert!(!reply_ok(&s.handle_line(
            r#"{"type":"move_joints","id":2,"angles_deg":[1,0,0,0,0,0],"speed":51}"#,
            t0
        )));
    }

    #[test]
    fn watchdog_stops_and_closes_the_gate() {
        let t0 = Instant::now();
        let mut s = enabled(t0);
        s.handle_line(r#"{"type":"ping"}"#, t0 + ms(400));
        s.tick(t0 + ms(900));
        assert!(s.gate_open(), "500 ms since the ping is not over the limit");
        s.tick(t0 + ms(901));
        assert!(!s.gate_open());
        assert!(s.gate_reason().starts_with("watchdog"));
        assert_eq!(s.arm().commands.last().map(String::as_str), Some("stop"));
    }

    #[test]
    fn stale_reading_with_gate_open_halts() {
        let t0 = Instant::now();
        let mut s = enabled(t0);
        s.arm.fail_reads = true;
        for k in 1..=3 {
            s.handle_line(r#"{"type":"ping"}"#, t0 + ms(100 * k));
            s.tick(t0 + ms(100 * k));
        }
        assert!(s.gate_open(), "300 ms old is still fresh");
        s.handle_line(r#"{"type":"ping"}"#, t0 + ms(400));
        s.tick(t0 + ms(400));
        assert!(!s.gate_open());
        assert!(s.gate_reason().contains("stale"));
    }

    #[test]
    fn stop_gets_through_a_closed_gate() {
        let t0 = Instant::now();
        let mut s = Server::new(DryRunArm::default(), Config::default());
        s.on_connect(t0);
        assert!(reply_ok(&s.handle_line(r#"{"type":"stop","id":7}"#, t0)));
        assert_eq!(s.arm().commands, vec!["stop"]);
    }

    #[test]
    fn disconnect_stops_even_with_the_gate_closed() {
        let t0 = Instant::now();
        let mut s = enabled(t0);
        s.handle_line(r#"{"type":"disable","id":2}"#, t0);
        s.on_disconnect("eof");
        assert_eq!(s.arm().commands, vec!["stop"]);
        assert_eq!(s.tick(t0 + ms(1000)), vec![]);
    }

    #[test]
    fn bad_line_is_answered_with_its_id() {
        let t0 = Instant::now();
        let mut s = enabled(t0);
        let r = s.handle_line(r#"{"type":"gripper","id":4,"value":101,"speed":50}"#, t0);
        assert!(matches!(
            &r[..],
            [Event::Reply {
                id: Some(4),
                ok: false,
                ..
            }]
        ));
        let r = s.handle_line("not json", t0);
        assert!(matches!(
            &r[..],
            [Event::Reply {
                id: None,
                ok: false,
                ..
            }]
        ));
    }

    #[test]
    fn state_reports_failed_read_as_null_with_the_last_age() {
        let t0 = Instant::now();
        let mut s = Server::new(DryRunArm::default(), Config::default());
        s.on_connect(t0);
        s.tick(t0);
        s.arm.fail_reads = true;
        match &s.tick(t0 + ms(100))[..] {
            [
                Event::State {
                    angles_deg: None,
                    read_age_ms: Some(100),
                    ..
                },
            ] => {}
            other => panic!("{other:?}"),
        }
    }
}
