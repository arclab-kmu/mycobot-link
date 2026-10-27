//! Frames byte-for-byte against pymycobot 4.0.7, recorded by
//! `tools/golden_frames.py` over a pty - what the SDK writes, not what its
//! source seems to say.

use mycobot_arm::frame::{self, FrameReader};
use serde_json::Value;

fn golden() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/golden/pymycobot-4.0.7.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

fn u8_arg(args: &Value, i: usize) -> u8 {
    args[i].as_u64().unwrap() as u8
}

#[test]
fn every_request_matches_pymycobot() {
    let g = golden();
    let requests = g["requests"].as_array().unwrap();
    assert!(requests.len() >= 10);
    for r in requests {
        let args = &r["args"];
        let ours = match r["call"].as_str().unwrap() {
            "get_angles" => frame::get_angles(),
            "send_angles" => {
                let q: Vec<f64> = args[0]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect();
                frame::send_angles(&q.try_into().unwrap(), u8_arg(args, 1))
            }
            "stop" => frame::stop(),
            "set_fresh_mode" => frame::set_fresh_mode(u8_arg(args, 0)),
            "set_gripper_value" => frame::set_gripper_value(u8_arg(args, 0), u8_arg(args, 1)),
            "get_gripper_value" => frame::get_gripper_value(),
            "is_power_on" => frame::is_power_on(),
            "get_error_information" => frame::get_error_information(),
            other => panic!("no Rust frame for {other}"),
        };
        assert_eq!(
            hex(&ours),
            r["frame"].as_str().unwrap(),
            "{} {}",
            r["call"],
            args
        );
    }
}

#[test]
fn get_angles_reply_decodes_as_pymycobot_reads_it() {
    let g = golden();
    for r in g["replies"].as_array().unwrap() {
        assert_eq!(r["call"], "get_angles");
        let mut reader = FrameReader::new();
        reader.push(&unhex(r["reply"].as_str().unwrap()));
        let got = frame::decode_angles(&reader.next_frame().unwrap()).unwrap();
        let want: Vec<f64> = r["result"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-9, "{got:?} vs {want:?}");
        }
    }
}
