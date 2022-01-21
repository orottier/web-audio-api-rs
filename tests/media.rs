use float_eq::assert_float_eq;
use web_audio_api::buffer::AudioBuffer;
use web_audio_api::context::{Context, OfflineAudioContext};
use web_audio_api::media::MediaElement;
use web_audio_api::node::{AudioControllableSourceNode, AudioNode, AudioScheduledSourceNode};
use web_audio_api::{SampleRate, RENDER_QUANTUM_SIZE};

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

struct SlowMedia {
    block: Arc<AtomicBool>,
    finished: Arc<AtomicBool>,
    sample_rate: SampleRate,
    value: f32,
}

impl Iterator for SlowMedia {
    type Item = Result<AudioBuffer, Box<dyn std::error::Error + Send + Sync>>;

    fn next(&mut self) -> Option<Self::Item> {
        // spin until we can emit
        while self.block.load(Ordering::SeqCst) {}
        // set blocking again
        self.block.store(true, Ordering::SeqCst);

        if self.finished.load(Ordering::SeqCst) {
            return None;
        }

        self.value += 1.;

        let samples = vec![vec![self.value; RENDER_QUANTUM_SIZE]];
        let buffer = AudioBuffer::from(samples, self.sample_rate);

        Some(Ok(buffer))
    }
}

#[test]
fn test_media_buffering() {
    let mut context = OfflineAudioContext::new(1, RENDER_QUANTUM_SIZE, SampleRate(44_100));

    let block = Arc::new(AtomicBool::new(true));
    let finished = Arc::new(AtomicBool::new(false));

    {
        let media = SlowMedia {
            block: block.clone(),
            finished: finished.clone(),
            sample_rate: SampleRate(44_100),
            value: 1.,
        };

        let element = MediaElement::new(media);
        let node = context.create_media_element_source(element);
        node.connect(&context.destination());
        node.set_loop(true); // test if silence is not included in buffer
        node.start();
    }

    // should be silent since the media stream did not yield any output
    let output = context.start_rendering();
    assert_float_eq!(
        output.get_channel_data(0),
        &[0.; RENDER_QUANTUM_SIZE][..],
        abs_all <= 0.
    );

    block.store(false, Ordering::SeqCst); // emit single chunk
    thread::sleep(Duration::from_millis(10)); // let buffer catch up

    // should contain output
    let output = context.start_rendering();
    assert_float_eq!(
        output.get_channel_data(0),
        &[2.; RENDER_QUANTUM_SIZE][..],
        abs_all <= 0.
    );

    // should be silent since the media stream did not yield any output
    let output = context.start_rendering();
    assert_float_eq!(
        output.get_channel_data(0),
        &[0.; RENDER_QUANTUM_SIZE][..],
        abs_all <= 0.
    );

    block.store(false, Ordering::SeqCst); // emit single chunk
    thread::sleep(Duration::from_millis(10)); // let buffer catch up

    // should contain output
    let output = context.start_rendering();
    assert_float_eq!(
        output.get_channel_data(0),
        &[3.; RENDER_QUANTUM_SIZE][..],
        abs_all <= 0.
    );

    finished.store(true, Ordering::SeqCst); // signal stream ended
    block.store(false, Ordering::SeqCst); // emit single chunk
    thread::sleep(Duration::from_millis(10)); // let buffer catch up

    // should contain previous output (looping)
    let output = context.start_rendering();
    assert_float_eq!(
        output.get_channel_data(0),
        &[2.; RENDER_QUANTUM_SIZE][..],
        abs_all <= 0.
    );
    let output = context.start_rendering();
    assert_float_eq!(
        output.get_channel_data(0),
        &[3.; RENDER_QUANTUM_SIZE][..],
        abs_all <= 0.
    );
    let output = context.start_rendering();
    assert_float_eq!(
        output.get_channel_data(0),
        &[2.; RENDER_QUANTUM_SIZE][..],
        abs_all <= 0.
    );
}

#[test]
fn test_media_seeking() {
    const SAMPLE_RATE: SampleRate = SampleRate(RENDER_QUANTUM_SIZE as u32); // 1 render quantum = 1 second
    let mut context = OfflineAudioContext::new(1, RENDER_QUANTUM_SIZE, SAMPLE_RATE);

    let block = Arc::new(AtomicBool::new(true));

    {
        let media = SlowMedia {
            block: block.clone(),
            finished: Arc::new(AtomicBool::new(false)),
            sample_rate: SAMPLE_RATE,
            value: 1.,
        };

        let element = MediaElement::new(media);
        let node = context.create_media_element_source(element);
        node.connect(&context.destination());
        node.seek(2.); // test seeking in combination with slow buffering
        node.start();
    }

    // should be silent since the media stream did not yield any output
    let output = context.start_rendering();
    assert_float_eq!(
        output.get_channel_data(0),
        &[0.; RENDER_QUANTUM_SIZE][..],
        abs_all <= 0.
    );

    block.store(false, Ordering::SeqCst); // emit single chunk
    thread::sleep(Duration::from_millis(10)); // let buffer catch up
    block.store(false, Ordering::SeqCst); // emit single chunk
    thread::sleep(Duration::from_millis(10)); // let buffer catch up
    block.store(false, Ordering::SeqCst); // emit single chunk
    thread::sleep(Duration::from_millis(10)); // let buffer catch up

    // should contain output, with first 2 values skipped
    let output = context.start_rendering();
    assert_float_eq!(
        output.get_channel_data(0),
        &[4.; RENDER_QUANTUM_SIZE][..],
        abs_all <= 0.
    );
}
