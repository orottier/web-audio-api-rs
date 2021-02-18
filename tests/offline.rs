use web_audio_api::context::AsBaseAudioContext;
use web_audio_api::context::OfflineAudioContext;
use web_audio_api::node::AudioScheduledSourceNode;
use web_audio_api::node::{AudioNode, OscillatorNode, OscillatorOptions, OscillatorType};
use web_audio_api::BUFFER_SIZE;

#[test]
fn test_start_stop() {
    let len = (BUFFER_SIZE * 4) as usize;
    let mut context = OfflineAudioContext::new(1, len, BUFFER_SIZE);
    assert_eq!(context.length(), len);

    let opts = OscillatorOptions {
        type_: OscillatorType::Square,
        frequency: 0., // constant signal
        ..Default::default()
    };
    let osc = OscillatorNode::new(&context, opts);
    osc.connect(&context.destination());

    osc.start_at(1.);
    osc.stop_at(3.);

    let output = context.start_rendering();

    // one chunk of silence, two chunks of signal, one chunk of silence
    let mut expected = vec![0.; BUFFER_SIZE as usize];
    expected.append(&mut vec![1.; 2 * BUFFER_SIZE as usize]);
    expected.append(&mut vec![0.; BUFFER_SIZE as usize]);

    assert_eq!(output, expected.as_slice());
}

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
        frequency: 0., // constant signal
        ..Default::default()
    };
    let osc = OscillatorNode::new(&context, opts);
    osc.connect(&delay);
    osc.start();

    let output = context.start_rendering();

    // two chunks of silence, two chunks of signal
    let mut expected = vec![0.; 2 * BUFFER_SIZE as usize];
    expected.append(&mut vec![1.; 2 * BUFFER_SIZE as usize]);

    assert_eq!(output, expected.as_slice());
}
