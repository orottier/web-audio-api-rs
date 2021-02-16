use std::fs::File;

use web_audio_api::buffer::{ChannelConfigOptions, ChannelCountMode, ChannelInterpretation};
use web_audio_api::context::AsBaseAudioContext;
use web_audio_api::context::AudioContext;
use web_audio_api::media::OggVorbisDecoder;
use web_audio_api::node::{AudioNode, OscillatorNode, OscillatorOptions, OscillatorType};
use web_audio_api::node::{
    AudioScheduledSourceNode, MediaElementAudioSourceNode, MediaElementAudioSourceNodeOptions,
};

fn main() {
    let media = OggVorbisDecoder::try_new(File::open("music.ogg").unwrap()).unwrap();
    let context = AudioContext::new();
    let opts = MediaElementAudioSourceNodeOptions {
        media,
        channel_config: ChannelConfigOptions {
            count: 2,
            mode: ChannelCountMode::Max,
            interpretation: ChannelInterpretation::Speakers,
        },
    };
    let osc = MediaElementAudioSourceNode::new(&context, opts);
    osc.connect(&context.destination());

    std::thread::sleep(std::time::Duration::from_secs(4));
}
