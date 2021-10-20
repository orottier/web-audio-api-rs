use float_eq::assert_float_eq;
use web_audio_api::context::AsBaseAudioContext;
use web_audio_api::context::OfflineAudioContext;
use web_audio_api::node::{
    AudioNode, AudioScheduledSourceNode, OscillatorNode, OscillatorOptions, OscillatorType,
};
use web_audio_api::BUFFER_SIZE;

#[test]
fn test_offline_render() {
    const LENGTH: usize = 555;

    // not a multiple of BUFFER_SIZE
    assert!(LENGTH % BUFFER_SIZE as usize != 0);

    let mut context = OfflineAudioContext::new(2, LENGTH, 44_100.);
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
    assert_eq!(output.length(), LENGTH);

    assert_float_eq!(
        output.get_channel_data(0)[..],
        &[-2.; LENGTH][..],
        ulps_all <= 0
    );
    assert_float_eq!(
        output.get_channel_data(1)[..],
        &[-2.; LENGTH][..],
        ulps_all <= 0
    );
}

#[test]
fn test_start_stop() {
    let len = (BUFFER_SIZE * 4) as usize;
    let mut context = OfflineAudioContext::new(1, len, BUFFER_SIZE as f32);
    assert_eq!(context.length(), len);

    {
        let opts = OscillatorOptions {
            type_: Some(OscillatorType::Square),
            frequency: Some(0.), // constant signal
            ..Default::default()
        };
        let osc = OscillatorNode::new(&context, Some(opts));
        osc.connect(&context.destination());

        osc.start_at(1.);
        osc.stop_at(3.);
    }

    let output = context.start_rendering();
    assert_eq!(output.number_of_channels(), 1);
    assert_eq!(output.length(), BUFFER_SIZE as usize * 4);

    let channel_data = &output.get_channel_data(0)[..];

    // one chunk of silence, two chunks of signal, one chunk of silence
    let mut expected = vec![0.; BUFFER_SIZE as usize];
    expected.append(&mut vec![1.; 2 * BUFFER_SIZE as usize]);
    expected.append(&mut vec![0.; BUFFER_SIZE as usize]);

    assert_eq!(channel_data, &expected[..]);
}

#[test]
fn test_delayed_constant_source() {
    let len = (BUFFER_SIZE * 4) as usize;
    let mut context = OfflineAudioContext::new(1, len, BUFFER_SIZE as f32);
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
    assert_eq!(output.length(), BUFFER_SIZE as usize * 4);

    let channel_data = &output.get_channel_data(0)[..];

    // two chunks of silence, two chunks of signal
    let mut expected = vec![0.; 2 * BUFFER_SIZE as usize];
    expected.append(&mut vec![1.; 2 * BUFFER_SIZE as usize]);

    assert_eq!(channel_data, &expected[..]);
}

#[test]
fn test_audio_param_graph() {
    let len = BUFFER_SIZE as usize;
    let mut context = OfflineAudioContext::new(1, len, BUFFER_SIZE as f32);
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
    assert_eq!(output.length(), BUFFER_SIZE as usize);

    let channel_data = &output.get_channel_data(0)[..];

    // expect output = 0.8 (input) * ( 0.5 (intrinsic gain) + 0.4 (via 2 constant source input) )
    let expected = vec![0.8 * 0.9; BUFFER_SIZE as usize];
    assert_eq!(channel_data, &expected[..]);
}

#[test]
fn test_listener() {
    let len = BUFFER_SIZE as usize;
    let mut context = OfflineAudioContext::new(1, len, BUFFER_SIZE as f32);

    {
        let listener1 = context.listener();
        let listener2 = context.listener();
        listener1.position_x().set_value(1.);
        listener2.position_y().set_value(2.);
    }

    let _ = context.start_rendering();

    let listener = context.listener();
    assert_float_eq!(listener.position_y().value(), 2., ulps <= 0);
    assert_float_eq!(listener.position_x().value(), 1., ulps <= 0);
}

#[test]
fn test_cycle() {
    let len = BUFFER_SIZE as usize;
    let mut context = OfflineAudioContext::new(1, len, 44_100.);

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
    assert_float_eq!(
        output.get_channel_data(0)[..],
        &[2.; BUFFER_SIZE as usize][..],
        ulps_all <= 0
    );
}
