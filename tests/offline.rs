use web_audio_api::context::AsBaseAudioContext;
use web_audio_api::context::OfflineAudioContext;
use web_audio_api::node::{
    AudioNode, AudioScheduledSourceNode, OscillatorNode, OscillatorOptions, OscillatorType,
};
use web_audio_api::{SampleRate, BUFFER_SIZE};

#[test]
fn test_offline_render() {
    const LENGTH: usize = 555;

    // not a multiple of BUFFER_SIZE
    assert!(LENGTH % BUFFER_SIZE as usize != 0);

    let mut context = OfflineAudioContext::new(2, LENGTH, SampleRate(44_100));
    assert_eq!(context.length(), LENGTH);

    {
        let constant1 = context.create_constant_source();
        constant1.offset().set_value(2.);
        constant1.connect(&context.destination());

        let constant2 = context.create_constant_source();
        constant2.offset().set_value(-4.);
        constant2.connect(&context.destination());
    }

    let output = context.start_rendering();
    assert_eq!(output.number_of_channels(), 2);
    assert_eq!(output.sample_len(), LENGTH);

    assert_eq!(output.channel_data(0).as_slice(), &[-2.; LENGTH]);
    assert_eq!(output.channel_data(1).as_slice(), &[-2.; LENGTH]);
}

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
    assert_eq!(output.number_of_channels(), 1);
    assert_eq!(output.sample_len(), BUFFER_SIZE as usize * 4);

    let channel_data = output.channel_data(0).as_slice();

    // one chunk of silence, two chunks of signal, one chunk of silence
    let mut expected = vec![0.; BUFFER_SIZE as usize];
    expected.append(&mut vec![1.; 2 * BUFFER_SIZE as usize]);
    expected.append(&mut vec![0.; BUFFER_SIZE as usize]);

    assert_eq!(channel_data, expected.as_slice());
}

#[test]
fn test_delayed_constant_source() {
    let len = (BUFFER_SIZE * 4) as usize;
    let mut context = OfflineAudioContext::new(1, len, SampleRate(BUFFER_SIZE));
    assert_eq!(context.length(), len);

    {
        let delay = context.create_delay(10.);
        delay.delay_time().set_value(2.);
        delay.connect(&context.destination());

        let source = context.create_constant_source();
        source.connect(&delay);
    }

    let output = context.start_rendering();
    assert_eq!(output.number_of_channels(), 1);
    assert_eq!(output.sample_len(), BUFFER_SIZE as usize * 4);

    let channel_data = output.channel_data(0).as_slice();

    // two chunks of silence, two chunks of signal
    let mut expected = vec![0.; 2 * BUFFER_SIZE as usize];
    expected.append(&mut vec![1.; 2 * BUFFER_SIZE as usize]);

    assert_eq!(channel_data, expected.as_slice());
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
    assert_eq!(output.number_of_channels(), 1);
    assert_eq!(output.sample_len(), BUFFER_SIZE as usize);

    let channel_data = output.channel_data(0).as_slice();

    // expect output = 0.8 (input) * ( 0.5 (intrinsic gain) + 0.4 (via 2 constant source input) )
    let expected = vec![0.8 * 0.9; BUFFER_SIZE as usize];
    assert_eq!(channel_data, expected.as_slice());
}

#[test]
fn test_listener() {
    let len = BUFFER_SIZE as usize;
    let mut context = OfflineAudioContext::new(1, len, SampleRate(BUFFER_SIZE));

    {
        let listener1 = context.listener();
        let listener2 = context.listener();
        listener1.position_x().set_value(1.);
        listener2.position_y().set_value(2.);
    }

    let _ = context.start_rendering();

    let listener = context.listener();
    assert_eq!(listener.position_y().value(), 2.);
    assert_eq!(listener.position_x().value(), 1.);
}

#[test]
fn test_cycle() {
    let len = BUFFER_SIZE as usize;
    let mut context = OfflineAudioContext::new(1, len, SampleRate(44_100));

    {
        let cycle1 = context.create_gain();
        cycle1.connect(&context.destination());

        let cycle2 = context.create_gain();
        cycle2.connect(&cycle1);

        // here we go
        cycle1.connect(&cycle2);

        let source_cycle = context.create_constant_source();
        source_cycle.offset().set_value(1.);
        source_cycle.connect(&cycle1);

        let other = context.create_constant_source();
        other.offset().set_value(2.);
        other.connect(&context.destination());
    }

    let output = context.start_rendering();
    // cycle should be muted, and other source should be processed
    assert_eq!(
        output.channel_data(0).as_slice(),
        &[2.; BUFFER_SIZE as usize]
    );
}
