# Rust Web Audio API

[![crates.io](https://img.shields.io/crates/v/web-audio-api.svg)](https://crates.io/crates/web-audio-api)
[![docs.rs](https://img.shields.io/docsrs/web-audio-api)](https://docs.rs/web-audio-api)

A pure Rust implementation of the Web Audio API, for use in non-browser contexts

## About the Web Audio API

The [Web Audio API](https://www.w3.org/TR/webaudio/)
([MDN docs](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API))
provides a powerful and versatile system for controlling audio on the Web,
allowing developers to choose audio sources, add effects to audio, create audio
visualizations, apply spatial effects (such as panning) and much more.

Our Rust implementation decouples the Web Audio API from the Web. You can now
use it in desktop apps, command line utilities, headless execution, etc.

## Example usage

```rust
use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

// set up the audio context with optimized settings for your hardware
let context = AudioContext::default();

// for background music, read from local file
let file = std::fs::File::open("samples/major-scale.ogg").unwrap();
let buffer = context.decode_audio_data_sync(file).unwrap();

// setup an AudioBufferSourceNode
let mut src = context.create_buffer_source();
src.set_buffer(buffer);
src.set_loop(true);

// create a biquad filter
let biquad = context.create_biquad_filter();
biquad.frequency().set_value(125.);

// connect the audio nodes
src.connect(&biquad);
biquad.connect(&context.destination());

// play the buffer
src.start();

// enjoy listening
loop { }
```

Check out the [docs](https://docs.rs/web-audio-api) for more info.

## Spec compliance

We have tried to stick to the official W3C spec as close as possible, but some
deviations could not be avoided:

- naming: snake_case instead of CamelCase
- getters/setters methods instead of exposed attributes
- introduced some namespacing
- AudioWorklet functionality is provided in a more rust-friendly way
- inheritance is modelled with traits
- ...

## Bindings

We provide NodeJS bindings to this library over at
<https://github.com/ircam-ismm/node-web-audio-api> so you can use this library
by simply writing native NodeJS code.

It is a work in progress, but eventually we should be able to run the official
[WebAudioAPI test harness](https://github.com/web-platform-tests/wpt/tree/master/webaudio)
and track our spec compliance improvements over time.

## Audio backends

By default, the [`cpal`](https://github.com/rustaudio/cpal) library is used for
cross platform audio I/O.

We offer [experimental support](https://github.com/orottier/web-audio-api-rs/issues/187) for the
[`cubeb`](https://github.com/mozilla/cubeb-rs) backend via the `cubeb` feature
flag. Please note that `cmake` must be installed locally in order to run
`cubeb`.

| Feature flag   | Backend        | Notes                                                   |
| -------------- | -------------- | ------------------------------------------------------- |
| cpal (default) | ALSA           |                                                         |
| cpal (default) | WASAPI         |                                                         |
| cpal (default) | CoreAudio      |                                                         |
| cpal (default) | Oboe (Android) |                                                         |
| cpal-jack      | JACK           |                                                         |
| cpal-asio      | ASIO           | see <https://github.com/rustaudio/cpal#asio-on-windows> |

| cubeb | PulseAudio | |
| cubeb | AudioUnit | |
| cubeb | WASAPI | |
| cubeb | OpenSL | |
| cubeb | AAudio | |
| cubeb | sndio | |
| cubeb | Sun | |
| cubeb | OSS | |

### Notes for Linux users

Using the library on Linux with the ALSA backend might lead to unexpected cranky sound with the default render size (i.e. 128 frames). In such cases, a simple workaround is to pass the `AudioContextLatencyCategory::Playback` latency hint when creating the audio context, which will increase the render size to 1024 frames:

```rs
let audio_context = AudioContext::new(AudioContextOptions {
    latency_hint: AudioContextLatencyCategory::Playback,
    ..AudioContextOptions::default()
});
```

For real-time and interactive applications where low latency is crucial, you should instead rely on the JACK backend provided by `cpal`. To that end you will need a running JACK server and build your application with the `cpal-jack` feature, e.g. `cargo run --release --features "cpal-jack" --example microphone`.

## Contributing

web-audio-api-rs welcomes contribution from everyone in the form of suggestions, bug reports,
pull requests, and feedback. 💛

If you need ideas for contribution, there are several ways to get started:

- Try out some of our examples (located in the `examples/` directory) and start
  building your own audio graphs
- Found a bug or have a feature request?
  [Submit an issue](https://github.com/orottier/web-audio-api-rs/issues/new)!
- Issues labeled with
  [good first issue](https://github.com/orottier/web-audio-api-rs/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc+label%3A%22good+first+issue%22)
  are relatively easy starter issues.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in web-audio-api-rs by you, shall be licensed as MIT, without any
additional terms or conditions.

## License

This project is licensed under the [MIT license].

[mit license]: https://github.com/orottier/web-audio-api-rs/blob/main/LICENSE

## Acknowledgements

The IR files used for HRTF spatialization are part of the LISTEN database
created by the EAC team from Ircam.
