use web_audio_api::context::AsBaseAudioContext;
use web_audio_api::context::OfflineAudioContext;
use web_audio_api::node::{
    AudioNode, AudioScheduledSourceNode, OscillatorNode, OscillatorOptions, OscillatorType,
};
use web_audio_api::{SampleRate, BUFFER_SIZE};

#[test]
fn test_start_stop() {
    let len = (BUFFER_SIZE * 4) as usize;
    let mut context = OfflineAudioContext::new(1, len, SampleRate(BUFFER_SIZE));
    assert_eq!(context.length(), len);

    {
        let opts = OscillatorOptions {
            type_: OscillatorType::Square,
            frequency: 0., // constant signal
            ..Default::default()
        };
        let osc = OscillatorNode::new(&context, opts);
        osc.connect(&context.destination());

        osc.start_at(1.);
        osc.stop_at(3.);
    }

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
    let mut context = OfflineAudioContext::new(1, len, SampleRate(44_100));
    assert_eq!(context.length(), len);

    {
        let delay = context.create_delay();
        delay.set_render_quanta(2);
        delay.connect(&context.destination());

        let source = context.create_constant_source();
        source.connect(&delay);
    }

    let output = context.start_rendering();

    // two chunks of silence, two chunks of signal
    let mut expected = vec![0.; 2 * BUFFER_SIZE as usize];
    expected.append(&mut vec![1.; 2 * BUFFER_SIZE as usize]);

    assert_eq!(output, expected.as_slice());
}

#[test]
fn test_audio_param_graph() {
    let len = BUFFER_SIZE as usize;
    let mut context = OfflineAudioContext::new(1, len, SampleRate(BUFFER_SIZE));
    {
        let gain = context.create_gain();
        gain.gain().set_value(0.5); // intrinsic value
        gain.connect(&context.destination());

        let source = context.create_constant_source();
        source.offset().set_value(0.8);
        source.connect(&gain);

        let param_input1 = context.create_constant_source();
        param_input1.offset().set_value(0.1);
        param_input1.connect(gain.gain());

        let param_input2 = context.create_constant_source();
        param_input2.offset().set_value(0.3);
        param_input2.connect(gain.gain());
    }

    let output = context.start_rendering();

    // expect output = 0.8 (input) * ( 0.5 (intrinsic gain) + 0.4 (via 2 constant source input) )
    let expected = vec![0.8 * 0.9; BUFFER_SIZE as usize];
    assert_eq!(output, expected.as_slice());
}
