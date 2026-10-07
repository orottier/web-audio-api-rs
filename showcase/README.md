# web-audio-api-rs showcase projects

These directories contain example projects using the web-audio-api-rs library.
To run them, simply `cd` into the directory and run `cargo run --release`.

You will find many more examples in the root `/examples` directory of this
repository. All examples that do not require external dependencies are put
there.

`raw_processor` is an external consumer of the `unstable_plumbing` feature. It
constructs a custom node/processor pair, creates an audio parameter, and sends a
control message before checking the offline output. It needs no audio device.
The feature exposes existing internals and may change in minor releases.
