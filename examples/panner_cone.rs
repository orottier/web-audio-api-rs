use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::node::AudioNode;
use web_audio_api::node::AudioScheduledSourceNode;

fn main() {
    env_logger::init();
    let context = AudioContext::new(None);

    // Create a friendly tone
    let tone = context.create_oscillator();
    tone.frequency().set_value_at_time(300.0f32, 0.);
    tone.start();

    // Connect tone > panner node > destination node
    let panner = context.create_panner();
    tone.connect(&panner);
    panner.connect(&context.destination());

    // The panner node is 1 unit in front of listener
    panner.position_y().set_value_at_time(1., 0.);
    // Reset default orientation
    panner.orientation_x().set_value(0.);

    // Panner rotates around their axis, every second
    let moving = context.create_oscillator();
    moving.start();
    moving.frequency().set_value_at_time(1., 0.);
    // Connect to x-orientation
    moving.connect(panner.orientation_x());
    // Connect to y-orientation with half phase delay
    let delay = context.create_delay(1.);
    delay.delay_time().set_value(0.25);
    delay.connect(panner.orientation_y());
    moving.connect(&delay);

    println!("The sound source spins around its axis every second");
    println!("Narrow cone - directional tone");
    panner.set_cone_inner_angle(30.);
    panner.set_cone_outer_angle(45.);
    std::thread::sleep(std::time::Duration::from_secs(4));

    println!("Medium cone");
    panner.set_cone_inner_angle(90.);
    panner.set_cone_outer_angle(270.);
    std::thread::sleep(std::time::Duration::from_secs(4));

    println!("Wide cone - near omnidirectional tone");
    panner.set_cone_outer_gain(0.5);
    panner.set_cone_inner_angle(180.);
    panner.set_cone_outer_angle(360.);
    std::thread::sleep(std::time::Duration::from_secs(4));
}
