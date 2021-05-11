use web_audio_api::buffer::{AudioBuffer, ChannelData};
use web_audio_api::context::{AsBaseAudioContext, OfflineAudioContext};
use web_audio_api::media::MediaElement;
use web_audio_api::node::{AudioControllableSourceNode, AudioNode, AudioScheduledSourceNode};
use web_audio_api::SampleRate;

fn main() {
    let len = 128 * 2048; // render a fixed number of frames
    let context = OfflineAudioContext::new(2, len, SampleRate(44_100));

    // create a big audio graph with all the nodes we know,
    // with each of the 10 pieces subtly different
    for i in 0..10 {
        {
            let channel = ChannelData::from(vec![i as f32 / 200.; 250]);
            let buf = AudioBuffer::from_channels(vec![channel; 2], SampleRate(96_000));
            let sequence = std::iter::repeat(buf).take(5);
            let media = sequence.map(|b| Ok(b));
            let element = MediaElement::new(media);

            let node = context.create_media_element_source(element);
            node.connect(&context.destination());
            node.set_loop(true);
            node.start();
        }

        let next_dest = {
            let add_l = context.create_constant_source();
            add_l.offset().set_value(i as f32 / 100.);
            let add_r = context.create_constant_source();
            add_r.offset().set_value(i as f32 / 50.);

            let split = context.create_channel_splitter(2);
            let merge = context.create_channel_merger(2);

            add_l.connect_at(&merge, 0, 0).unwrap();
            add_r.connect_at(&merge, 0, 1).unwrap();
            split.connect_at(&merge, 0, 0).unwrap();
            split.connect_at(&merge, 1, 1).unwrap();

            merge.connect(&context.destination());

            split
        };

        let next_dest = {
            let delay = context.create_delay(1.);
            delay.delay_time().set_value(i as f32 / 100.);
            delay.connect(&next_dest);

            delay
        };

        // Create a friendly tone
        let tone = context.create_oscillator();
        tone.frequency().set_value_at_time(300.0f32 + i as f32, 0.);
        tone.start();

        // Connect tone > panner node > destination node
        let panner = context.create_panner();
        tone.connect(&panner);
        panner.connect(&next_dest);

        // The panner node is 1 unit in front of listener
        panner.position_z().set_value_at_time(1. + i as f32, 0.);

        // And sweeps 10 units left to right, every second
        let moving = context.create_oscillator();
        moving.start();
        moving.frequency().set_value_at_time(i as f32, 0.);
        let gain = context.create_gain();
        gain.gain().set_value_at_time(10. + i as f32, 0.);
        moving.connect(&gain);
        gain.connect(panner.position_x());
    }

    let mut context = context;
    let output = context.start_rendering();
    assert_eq!(output.sample_len(), len);
}
