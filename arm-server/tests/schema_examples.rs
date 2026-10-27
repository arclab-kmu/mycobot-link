//! The Rust types must take every example the Python side validates against
//! `schema/messages.schema.json`, and write the arm's examples back unchanged.

use mycobot_arm::protocol::{Event, Request};
use serde_json::Value;

fn examples() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../schema/examples.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn brain_examples_parse_as_requests() {
    for ex in examples()["brain_to_arm"].as_array().unwrap() {
        let parsed: Result<Request, _> = serde_json::from_value(ex.clone());
        assert!(parsed.is_ok(), "{ex}: {parsed:?}");
    }
}

#[test]
fn arm_examples_round_trip_through_events() {
    for ex in examples()["arm_to_brain"].as_array().unwrap() {
        let event: Event =
            serde_json::from_value(ex.clone()).unwrap_or_else(|e| panic!("{ex}: {e}"));
        assert_eq!(&serde_json::to_value(&event).unwrap(), ex);
    }
}
