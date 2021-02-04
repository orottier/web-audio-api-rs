use web_audio_api::context::AsBaseAudioContext;
use web_audio_api::context::OfflineAudioContext;
use web_audio_api::node::{AudioNode, OscillatorNode, OscillatorOptions, OscillatorType};
use web_audio_api::BUFFER_SIZE;

#[test]
fn test_delayed_constant_source() {
    let len = (BUFFER_SIZE * 4) as usize;
    let mut context = OfflineAudioContext::new(1, len, 44_100);
    assert_eq!(context.length(), len);

    let delay = context.create_delay();
    delay.set_render_quanta(2);
    delay.connect(&context.destination());

    let opts = OscillatorOptions {
        type_: OscillatorType::Square,
        frequency: 0, // constant signal
    };
    let osc = OscillatorNode::new(&context, opts);
    osc.connect(&delay);

    let output = context.start_rendering();

    // two chunks of silence, two chunks of signal
    let mut expected = vec![0.; 2 * BUFFER_SIZE as usize];
    expected.append(&mut vec![1.; 2 * BUFFER_SIZE as usize]);

    assert_eq!(output, expected.as_slice());
}
