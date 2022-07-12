use float_eq::assert_float_eq;
use std::sync::Arc;
use web_audio_api::context::{BaseAudioContext, OfflineAudioContext};

pub fn main() {
    // Setup an OfflineAudioContext for decoding, the actual length and channel count do not matter
    let context = OfflineAudioContext::new(2, 100, 44100.);

    // We need shared ownership of the context because the Rust compiler cannot infer that our
    // threads will not outlive the main thread.
    //
    // TODO update after Rust 1.63 to use scoped threads
    let context = Arc::new(context);

    // Setup five threads and let them decode audio buffers concurrently
    let handles = (0..5).into_iter().map(|_| {
        let context = context.clone();
        std::thread::spawn(move || {
            let file = std::fs::File::open("samples/sample.wav").unwrap();
            let audio_buffer = context.decode_audio_data_sync(file).unwrap();
            audio_buffer
        })
    });

    // Await the concurrent threads and validate their results
    handles.for_each(|handle| {
        let audio_buffer = handle.join().unwrap();

        assert_eq!(audio_buffer.sample_rate(), context.sample_rate());
        assert_eq!(audio_buffer.length(), 142_187);
        assert_eq!(audio_buffer.number_of_channels(), 2);
        assert_float_eq!(audio_buffer.duration(), 3.224, abs_all <= 0.001);
    });
}
