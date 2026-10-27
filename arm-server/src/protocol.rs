//! JSON-lines messages. `schema/messages.schema.json` is the reference;
//! `tests/schema_examples.rs` holds these types to `schema/examples.json`.

use serde::{Deserialize, Serialize};

pub const PROTOCOL_VERSION: u32 = 1;

/// brain -> arm.
///
/// Unknown fields are refused: with no optional fields, a misspelt one is a
/// bug in the brain, and silently dropping it is how a bug moves an arm.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Request {
    /// `{}` rather than a unit variant: serde lets a tagged unit variant
    /// through with any extra fields, deny_unknown_fields or not.
    Ping {},
    Enable {
        id: u64,
    },
    Disable {
        id: u64,
    },
    Stop {
        id: u64,
    },
    MoveJoints {
        id: u64,
        angles_deg: Vec<f64>,
        speed: u32,
    },
    Gripper {
        id: u64,
        value: u32,
        speed: u32,
    },
}

/// arm -> brain.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Event {
    Hello {
        protocol: u32,
        server: String,
        live: bool,
        watchdog_ms: u64,
        state_hz: f64,
    },
    Reply {
        id: Option<u64>,
        ok: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    State {
        seq: u64,
        angles_deg: Option<[f64; 6]>,
        read_age_ms: Option<u64>,
        gate_open: bool,
        gate_reason: String,
        live: bool,
    },
}

impl Event {
    pub fn ok(id: u64) -> Self {
        Event::Reply {
            id: Some(id),
            ok: true,
            error: None,
        }
    }

    pub fn rejected(id: Option<u64>, error: impl Into<String>) -> Self {
        Event::Reply {
            id,
            ok: false,
            error: Some(error.into()),
        }
    }
}

/// Parse one line. On failure, also return the `id` if the line carried one,
/// so the brain's pending request fails now instead of timing out.
pub fn parse_request(line: &str) -> Result<Request, (Option<u64>, String)> {
    let value: serde_json::Value =
        serde_json::from_str(line).map_err(|e| (None, format!("bad json: {e}")))?;
    let id = value.get("id").and_then(serde_json::Value::as_u64);
    serde_json::from_value(value).map_err(|e| (id, format!("bad request: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn misspelt_field_is_refused_with_its_id() {
        let err =
            parse_request(r#"{"type":"move_joints","id":9,"angle_deg":[0,0,0,0,0,0],"speed":20}"#)
                .unwrap_err();
        assert_eq!(err.0, Some(9));
    }

    #[test]
    fn fractional_speed_is_refused() {
        assert!(parse_request(r#"{"type":"gripper","id":1,"value":10,"speed":20.5}"#).is_err());
    }

    #[test]
    fn ping_with_extra_fields_is_refused() {
        assert!(parse_request(r#"{"type":"ping","id":3}"#).is_err());
    }
}
