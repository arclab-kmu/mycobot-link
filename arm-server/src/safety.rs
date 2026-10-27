//! Limits, ported from `mycobot-control/src/mycobot_control/safety/limits.py`.
//!
//! Units are the SDK's: degrees. Keep the numbers in step with that file; the
//! reasoning behind each one is recorded there.

pub const N_JOINTS: usize = 6;

/// pymycobot 4.0.7 `robot_info.py` "MyCobot280" angles_min / angles_max, which
/// agree with the vendor URDF. TODO(unverified): the controller's own values
/// (`get_joint_min_angle` / `get_joint_max_angle`).
pub const JOINT_LIMITS_DEG: [(f64, f64); N_JOINTS] = [
    (-168.0, 168.0), // J1
    (-140.0, 140.0), // J2
    (-150.0, 150.0), // J3
    (-150.0, 150.0), // J4
    (-155.0, 160.0), // J5
    (-180.0, 180.0), // J6
];

/// Hobby-class servos and a source-constant table: never command the last few degrees.
pub const JOINT_LIMIT_MARGIN_DEG: f64 = 3.0;

/// Largest jump from the *measured* angles. Refused, not clamped: a jump this
/// large usually means rad passed as deg, or a stale reading.
pub const MAX_COMMAND_DELTA_DEG: f64 = 20.0;

pub const SPEED_MIN: u32 = 1;
pub const SPEED_MAX: u32 = 100;
/// Above this `mycobot-control` asks for `--allow-fast`. A choice, not a vendor number.
pub const FAST_SPEED: u32 = 50;

pub const GRIPPER_MAX: u32 = 100;

pub fn check_joint_target(angles_deg: &[f64]) -> Result<[f64; N_JOINTS], String> {
    let q: [f64; N_JOINTS] = angles_deg
        .try_into()
        .map_err(|_| format!("need {N_JOINTS} joint angles, got {}", angles_deg.len()))?;
    if let Some(a) = q.iter().find(|a| !a.is_finite()) {
        return Err(format!("joint angles must be finite, got {a}"));
    }
    let bad: Vec<String> = q
        .iter()
        .zip(JOINT_LIMITS_DEG)
        .enumerate()
        .filter(|(_, (a, (lo, hi)))| {
            !(lo + JOINT_LIMIT_MARGIN_DEG..=hi - JOINT_LIMIT_MARGIN_DEG).contains(*a)
        })
        .map(|(i, (a, (lo, hi)))| {
            format!(
                "J{}={a:+.1} (limit {lo:+.0}..{hi:+.0}, margin {JOINT_LIMIT_MARGIN_DEG:.0})",
                i + 1
            )
        })
        .collect();
    if bad.is_empty() {
        Ok(q)
    } else {
        Err(format!("joint target outside limits: {}", bad.join(", ")))
    }
}

/// Returns the largest per-joint delta, or why it is too large.
pub fn check_command_delta(
    target: &[f64; N_JOINTS],
    measured: &[f64; N_JOINTS],
    max_delta_deg: f64,
) -> Result<f64, String> {
    let (worst_joint, worst) = target
        .iter()
        .zip(measured)
        .map(|(t, m)| (t - m).abs())
        .enumerate()
        .fold(
            (0, 0.0_f64),
            |best, (i, d)| if d > best.1 { (i, d) } else { best },
        );
    if worst > max_delta_deg {
        return Err(format!(
            "J{} is {worst:.1} deg from the measured angle (max {max_delta_deg:.1})",
            worst_joint + 1
        ));
    }
    Ok(worst)
}

pub fn check_speed(speed: u32, max: u32) -> Result<u8, String> {
    if (SPEED_MIN..=max.min(SPEED_MAX)).contains(&speed) {
        Ok(speed as u8)
    } else {
        Err(format!(
            "speed {speed} outside [{SPEED_MIN}, {}]",
            max.min(SPEED_MAX)
        ))
    }
}

pub fn check_gripper(value: u32) -> Result<u8, String> {
    if value <= GRIPPER_MAX {
        Ok(value as u8)
    } else {
        Err(format!("gripper value {value} outside [0, {GRIPPER_MAX}]"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn margin_applies_at_both_ends() {
        assert!(check_joint_target(&[165.0, 0.0, 0.0, 0.0, 0.0, 0.0]).is_ok());
        assert!(check_joint_target(&[165.1, 0.0, 0.0, 0.0, 0.0, 0.0]).is_err());
        assert!(check_joint_target(&[0.0, 0.0, 0.0, 0.0, -152.1, 0.0]).is_err());
    }

    #[test]
    fn wrong_length_and_nan_are_refused() {
        assert!(check_joint_target(&[0.0; 5]).is_err());
        assert!(check_joint_target(&[0.0, f64::NAN, 0.0, 0.0, 0.0, 0.0]).is_err());
    }

    #[test]
    fn delta_names_the_worst_joint() {
        let err =
            check_command_delta(&[0.0, 25.0, 0.0, 0.0, 0.0, 0.0], &[0.0; 6], 20.0).unwrap_err();
        assert!(err.starts_with("J2 is 25.0"), "{err}");
        assert_eq!(
            check_command_delta(&[0.0, 20.0, 0.0, 0.0, 0.0, 0.0], &[0.0; 6], 20.0),
            Ok(20.0)
        );
    }

    #[test]
    fn speed_cap_never_exceeds_the_sdk_range() {
        assert!(check_speed(0, 50).is_err());
        assert!(check_speed(51, 50).is_err());
        assert!(check_speed(101, 200).is_err());
        assert_eq!(check_speed(50, 50), Ok(50));
    }
}
