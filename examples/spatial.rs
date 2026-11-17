use std::f32::consts::TAU;
use std::io::BufRead;
use std::sync::mpsc;
use std::time::Duration;
use web_audio_api::context::{
    AudioContext, AudioContextLatencyCategory, AudioContextOptions, BaseAudioContext,
};
use web_audio_api::node::{
    AudioNode, AudioScheduledSourceNode, PannerNode, PannerOptions, PanningModelType,
};

// PannerNode example
//
// `cargo run --release --example spatial`
//
// If you are on Linux and use ALSA as audio backend backend, you might want to run
// the example with the `WEB_AUDIO_LATENCY=playback ` env variable which will
// increase the buffer size to 1024
//
// `WEB_AUDIO_LATENCY=playback cargo run --release --example spatial`
fn main() {
    env_logger::init();

    let latency_hint = match std::env::var("WEB_AUDIO_LATENCY").as_deref() {
        Ok("playback") => AudioContextLatencyCategory::Playback,
        _ => AudioContextLatencyCategory::default(),
    };

    let context = AudioContext::new(AudioContextOptions {
        latency_hint,
        ..AudioContextOptions::default()
    });

    // Create looping 'siren' sound
    let file = std::fs::File::open("samples/siren.mp3").unwrap();
    let buffer = context.decode_audio_data_sync(file).unwrap();
    let mut tone = context.create_buffer_source();
    tone.set_buffer(buffer);
    tone.set_loop(true);

    // Set up a panner with HRTF panning
    let opts = PannerOptions {
        panning_model: PanningModelType::HRTF,
        ..PannerOptions::default()
    };

    let mut panner = PannerNode::new(&context, opts);
    tone.connect(&panner);
    panner.connect(&context.destination());

    // Wind around a sphere, rising and falling twice per revolution. Viewed from
    // the front, this traces a figure eight. Normalize to keep distance gain steady.
    const PERIOD: f64 = 16.;
    const CURVE_STEPS: usize = 1024;
    const ANCHORS: [&str; 8] = [
        "Front",
        "Top front-right",
        "Full right",
        "Bottom back-right",
        "Behind",
        "Top back-left",
        "Full left",
        "Bottom front-left",
    ];
    let mut curves = [Vec::new(), Vec::new(), Vec::new()];
    for i in 0..=CURVE_STEPS {
        // Make the closing point identical to the first for a seamless repeat.
        let phase = TAU * (i % CURVE_STEPS) as f32 / CURVE_STEPS as f32;
        let position = orbit_position(phase);
        for (curve, value) in curves.iter_mut().zip(position) {
            curve.push(value);
        }
    }

    println!("Siren follows a 3D figure eight around you: one lap every {PERIOD} seconds.");
    println!("Use headphones. HRTF enabled; press <Enter> to toggle, or q then <Enter> to quit.");

    // Keep keyboard input from blocking motion scheduling and direction announcements.
    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        for line in std::io::stdin().lock().lines() {
            let Ok(line) = line else { break };
            if sender.send(line).is_err() {
                break;
            }
        }
    });

    let start_time = context.current_time() + 0.1;
    let mut next_cycle = start_time;
    let mut next_anchor = 0;
    let mut hrtf = true;
    tone.start_at(start_time);

    loop {
        let now = context.current_time();
        // Schedule on the audio clock, a full lap ahead, so movement remains smooth
        // even if the terminal thread wakes up late. Curves meet without overlap.
        while next_cycle < now + PERIOD {
            for (param, curve) in [
                panner.position_x(),
                panner.position_y(),
                panner.position_z(),
            ]
            .iter()
            .zip(&curves)
            {
                param.set_value_curve_at_time(curve, next_cycle, PERIOD);
            }
            next_cycle += PERIOD;
        }

        while now >= start_time + next_anchor as f64 * PERIOD / ANCHORS.len() as f64 {
            let index = next_anchor % ANCHORS.len();
            let [x, y, z] = orbit_position(TAU * index as f32 / ANCHORS.len() as f32);
            println!(
                "Lap {} | {:<18} | x={x:5.1}, y={y:5.1}, z={z:5.1}",
                next_anchor / ANCHORS.len() + 1,
                ANCHORS[index]
            );
            next_anchor += 1;
        }

        match receiver.try_recv() {
            Ok(line) if line.trim().eq_ignore_ascii_case("q") => break,
            Ok(_) => {
                hrtf = !hrtf;
                panner.set_panning_model(if hrtf {
                    PanningModelType::HRTF
                } else {
                    PanningModelType::EqualPower
                });
                println!("PanningMode: {:?}", panner.panning_model());
            }
            Err(mpsc::TryRecvError::Disconnected) => break,
            Err(mpsc::TryRecvError::Empty) => {}
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn orbit_position(phase: f32) -> [f32; 3] {
    let x = phase.sin();
    let y = 0.8 * (2. * phase).sin();
    let z = -phase.cos(); // Web Audio's default listener faces -Z.
    let scale = 10. / (x * x + y * y + z * z).sqrt();
    [x * scale, y * scale, z * scale]
}
