use float_eq::assert_float_eq;
use web_audio_api::buffer::{AudioBuffer, ChannelData};
use web_audio_api::context::AsBaseAudioContext;
use web_audio_api::context::OfflineAudioContext;
use web_audio_api::media::MediaElement;
use web_audio_api::node::{AudioControllableSourceNode, AudioNode, AudioScheduledSourceNode};
use web_audio_api::{SampleRate, BUFFER_SIZE};

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
    type Item = Result<AudioBuffer, Box<dyn std::error::Error + Send>>;

    fn next(&mut self) -> Option<Self::Item> {
        // spin until we can emit
        while self.block.load(Ordering::SeqCst) {}
        // set blocking again
        self.block.store(true, Ordering::SeqCst);

        if self.finished.load(Ordering::SeqCst) {
            return None;
        }

        self.value += 1.;

        let channel_data = ChannelData::from(vec![self.value; BUFFER_SIZE as usize]);
        let buffer = AudioBuffer::from_channels(vec![channel_data], self.sample_rate);

        Some(Ok(buffer))
    }
}

#[test]
fn test_media_buffering() {
    const LENGTH: usize = BUFFER_SIZE as usize;
    let mut context = OfflineAudioContext::new(1, LENGTH, SampleRate(44_100));

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
        output.channel_data(0).as_slice(),
        &[0.; LENGTH][..],
        ulps_all <= 0
    );

    block.store(false, Ordering::SeqCst); // emit single chunk
    thread::sleep(Duration::from_millis(10)); // let buffer catch up

    // should contain output
    let output = context.start_rendering();
    assert_float_eq!(
        output.channel_data(0).as_slice(),
        &[2.; LENGTH][..],
        ulps_all <= 0
    );

    // should be silent since the media stream did not yield any output
    let output = context.start_rendering();
    assert_float_eq!(
        output.channel_data(0).as_slice(),
        &[0.; LENGTH][..],
        ulps_all <= 0
    );

    block.store(false, Ordering::SeqCst); // emit single chunk
    thread::sleep(Duration::from_millis(10)); // let buffer catch up

    // should contain output
    let output = context.start_rendering();
    assert_float_eq!(
        output.channel_data(0).as_slice(),
        &[3.; LENGTH][..],
        ulps_all <= 0
    );

    finished.store(true, Ordering::SeqCst); // signal stream ended
    block.store(false, Ordering::SeqCst); // emit single chunk
    thread::sleep(Duration::from_millis(10)); // let buffer catch up

    // should contain previous output (looping)
    let output = context.start_rendering();
    assert_float_eq!(
        output.channel_data(0).as_slice(),
        &[2.; LENGTH][..],
        ulps_all <= 0
    );
    let output = context.start_rendering();
    assert_float_eq!(
        output.channel_data(0).as_slice(),
        &[3.; LENGTH][..],
        ulps_all <= 0
    );
    let output = context.start_rendering();
    assert_float_eq!(
        output.channel_data(0).as_slice(),
        &[2.; LENGTH][..],
        ulps_all <= 0
    );
}

#[test]
fn test_media_seeking() {
    const LENGTH: usize = BUFFER_SIZE as usize;
    const SAMPLE_RATE: SampleRate = SampleRate(BUFFER_SIZE); // 1 render quantum = 1 second
    let mut context = OfflineAudioContext::new(1, LENGTH, SAMPLE_RATE);

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
        output.channel_data(0).as_slice(),
        &[0.; LENGTH][..],
        ulps_all <= 0
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
        output.channel_data(0).as_slice(),
        &[4.; LENGTH][..],
        ulps_all <= 0
    );
}
