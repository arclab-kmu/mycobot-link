//! `mycobot-arm`: connect out to the brain, serve its requests, stop when it goes quiet.

use std::io::{self, IsTerminal, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use clap::Parser;
use mycobot_arm::arm::{Arm, DryRunArm, SerialArm};
use mycobot_arm::lines::LineBuffer;
use mycobot_arm::protocol::Event;
use mycobot_arm::safety;
use mycobot_arm::server::{Config, Server};

#[derive(Parser, Debug)]
#[command(
    version,
    about = "myCobot 280 JN arm server for mycobot-link (dry-run unless --live)"
)]
struct Args {
    /// Brain address, HOST:PORT. The brain binds; this side connects.
    #[arg(long)]
    brain: String,

    /// Drive the real arm over the UART. Without it nothing leaves this process.
    #[arg(long)]
    live: bool,

    /// Skip the typed `yes` that --live asks for (needed when stdin is not a TTY).
    #[arg(long)]
    assume_yes: bool,

    #[arg(long, default_value = "/dev/ttyTHS1")]
    serial: String,

    #[arg(long, default_value_t = 1_000_000)]
    baud: u32,

    /// Wait this long for a get_angles reply. pymycobot waits 500 ms; the
    /// state loop cannot, so this is short. TODO(unverified): measured reply time.
    #[arg(long, default_value_t = 50)]
    reply_timeout_ms: u64,

    #[arg(long, default_value_t = 500)]
    watchdog_ms: u64,

    #[arg(long, default_value_t = 10.0)]
    state_hz: f64,

    #[arg(long, default_value_t = 300)]
    max_read_age_ms: u64,

    #[arg(long, default_value_t = safety::FAST_SPEED)]
    max_speed: u32,
}

fn log(msg: impl AsRef<str>) {
    eprintln!("[mycobot-arm] {}", msg.as_ref());
}

fn confirm_live(args: &Args) -> Result<(), String> {
    if args.assume_yes {
        return Ok(());
    }
    if !io::stdin().is_terminal() {
        return Err("--live needs a typed `yes` on a terminal, or --assume-yes".into());
    }
    eprint!(
        "--live: commands from {} will move the arm on {}. Type yes: ",
        args.brain, args.serial
    );
    let mut answer = String::new();
    io::stdin()
        .read_line(&mut answer)
        .map_err(|e| e.to_string())?;
    if answer.trim() == "yes" {
        Ok(())
    } else {
        Err("not confirmed".into())
    }
}

fn send(stream: &mut TcpStream, events: Vec<Event>) -> io::Result<()> {
    for event in events {
        let mut line = serde_json::to_vec(&event).map_err(io::Error::other)?;
        line.push(b'\n');
        stream.write_all(&line)?;
    }
    Ok(())
}

/// One connection, until it drops or Ctrl-C.
fn run_session<A: Arm>(
    stream: TcpStream,
    server: &mut Server<A>,
    quit: &AtomicBool,
) -> io::Result<String> {
    stream.set_nodelay(true)?;
    // Short, so the loop keeps ticking the watchdog and the state clock.
    stream.set_read_timeout(Some(Duration::from_millis(10)))?;
    let mut reader = stream.try_clone()?;
    let mut writer = stream;
    let mut lines = LineBuffer::new();
    let mut buf = [0u8; 4096];
    send(&mut writer, server.on_connect(Instant::now()))?;
    loop {
        if quit.load(Ordering::SeqCst) {
            return Ok("interrupted".into());
        }
        match reader.read(&mut buf) {
            Ok(0) => return Ok("brain closed the connection".into()),
            Ok(n) => {
                for line in lines.push(&buf[..n]).map_err(io::Error::other)? {
                    send(&mut writer, server.handle_line(&line, Instant::now()))?;
                }
            }
            Err(e)
                if matches!(
                    e.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                ) => {}
            Err(e) => return Err(e),
        }
        send(&mut writer, server.tick(Instant::now()))?;
    }
}

fn serve<A: Arm>(args: &Args, arm: A, quit: &AtomicBool) {
    let cfg = Config {
        watchdog: Duration::from_millis(args.watchdog_ms),
        state_period: Duration::from_secs_f64(1.0 / args.state_hz),
        max_read_age: Duration::from_millis(args.max_read_age_ms),
        max_speed: args.max_speed,
        max_delta_deg: safety::MAX_COMMAND_DELTA_DEG,
    };
    let mut server = Server::new(arm, cfg);
    while !quit.load(Ordering::SeqCst) {
        let addr = match args.brain.to_socket_addrs().map(|mut a| a.next()) {
            Ok(Some(addr)) => addr,
            Ok(None) | Err(_) => {
                log(format!("cannot resolve {}", args.brain));
                thread::sleep(Duration::from_secs(1));
                continue;
            }
        };
        match TcpStream::connect_timeout(&addr, Duration::from_secs(2)) {
            Ok(stream) => {
                log(format!("connected to {addr}"));
                let why = match run_session(stream, &mut server, quit) {
                    Ok(why) => why,
                    Err(e) => e.to_string(),
                };
                log(format!("session ended: {why}"));
                server.on_disconnect(&why);
            }
            Err(e) => log(format!("connect {addr}: {e}")),
        }
        if !quit.load(Ordering::SeqCst) {
            thread::sleep(Duration::from_secs(1));
        }
    }
    server.shutdown();
}

fn main() -> ExitCode {
    let args = Args::parse();
    if !(args.state_hz > 0.0 && args.state_hz <= 100.0) {
        log("--state-hz must be in (0, 100]");
        return ExitCode::FAILURE;
    }
    let quit = Arc::new(AtomicBool::new(false));
    {
        let quit = quit.clone();
        if let Err(e) = ctrlc::set_handler(move || quit.store(true, Ordering::SeqCst)) {
            log(format!("cannot install the Ctrl-C handler: {e}"));
            return ExitCode::FAILURE;
        }
    }
    if args.live {
        if let Err(e) = confirm_live(&args) {
            log(format!("refusing --live: {e}"));
            return ExitCode::FAILURE;
        }
        let arm = match SerialArm::open(
            &args.serial,
            args.baud,
            Duration::from_millis(args.reply_timeout_ms),
        ) {
            Ok(arm) => arm,
            Err(e) => {
                log(format!("cannot open {}: {e}", args.serial));
                return ExitCode::FAILURE;
            }
        };
        log(format!("LIVE on {} @ {}", args.serial, args.baud));
        serve(&args, arm, &quit);
    } else {
        log("dry-run: nothing is sent to the arm (use --live for the UART)");
        serve(&args, DryRunArm::default(), &quit);
    }
    log("stopped");
    ExitCode::SUCCESS
}
