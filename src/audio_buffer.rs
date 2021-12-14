use std::rc::Rc;
use std::sync::Arc;

use crate::SampleRate;

// @todo - put it there, but should leave in `BaseAudioContext`
// let's just go with Wav files for now
//
// @note - what about using https://github.com/pdeljanov/Symphonia? seems quite
// complete and efficient
// @note - should also be async, but that's not a big deal neither for now
pub fn decode_audio_data(file: std::fs::File) -> AudioBuffer{
    let buf_reader = std::io::BufReader::new(file);
    let mut reader = hound::WavReader::new(buf_reader).unwrap();
    let hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample, // should probably use that (or some higher level library)
        sample_format,
    } = reader.spec();
    // shadow `channels`, this name is too usefull to be wasted here :)
    let number_of_channels = channels as usize;
    // badly named, there is not time information here as this is the number of
    // samples per channel, means nothing without sample_rate
    let length = reader.duration() as usize;

    // println!("channels {:?}", channels);
    // println!("length {:?}", length);
    // println!("sample_rate {:?}", sample_rate);
    // println!("bits_per_sample {:?}", bits_per_sample);
    // println!("sample_format {:?}", sample_format);

    // @note - we use this intermediary type (e.g. without any `Arc`), because we
    // need `mut` access and we don't want to go into `Mutex` as nothing is mutable
    // after this step. `AudioBufferChannel::from` should be the one that minimize
    // memory allocation if possible
    let mut decoded: Vec::<Vec<f32>> = Vec::with_capacity(number_of_channels);
    // init each channel with an empty Vec
    for _ in 0..number_of_channels {
        decoded.push(Vec::<f32>::with_capacity(length));
    }

    // @note - hound retrieve interleaved values
    // cf. https://docs.rs/hound/latest/hound/struct.WavReader.html#method.samples
    match sample_format {
        hound::SampleFormat::Int => {
            // channel are interleaved, so we need to de-interleave
            let mut channel_number = 0;

            // we should probably match `bit_per_sample` here and just create a
            // `max` variable to scale PCM data (e.g. `i[bits_per_sample]::MAX`
            if bits_per_sample == 16 {
                for sample in reader.samples::<i16>() {
                    let s = sample.unwrap() as f32 / i16::MAX as f32;
                    assert!(s.abs() < 1.);
                    decoded[channel_number].push(s);
                    // next sample belongs to the next channel
                    channel_number = (channel_number + 1) % number_of_channels;
                }
            } else {
                panic!("bits_per_sample {:?} not implemented", bits_per_sample);
            }
        }
        hound::SampleFormat::Float => { // this one is not tested
            let mut channel_number = 0;

            for sample in reader.samples::<f32>() {
                let s = sample.unwrap();
                decoded[channel_number].push(s);
                // next sample belongs to the next channel
                channel_number = (channel_number + 1) % number_of_channels;
            }
        }
    }

    assert_eq!(decoded[0].len(), length); // duration is ok
    // println!("decoded length: {} - input length: {}", decoded[0].len(), length);

    // [spec] Take the result, representing the decoded linear PCM audio data,
    // and resample it to the sample-rate of the BaseAudioContext if it is
    // different from the sample-rate of audioData.

    // @todo - resample if needed
    // if (sample_rate != self.sample_rate()) {
    //     // @todo - resample
    //     // cf. https://github.com/HEnquist/rubato/blob/master/examples/fftfixedin64.rs
    //     // already used in waveshaper
    // }

    AudioBuffer {
        number_of_channels,
        length,
        // @TODO - this is wrong, should be the AudioContext.sampleRate
        sample_rate: SampleRate(sample_rate),
        internal_data: AudioBufferData::from(decoded),
    }
}

// `AudioBufferData` is basically a simple matrix representing an aligned,
// de-interleaved "in memory" audio asset (i.e. a complete sound file), i.e.:
// 0 (left) : [0.0, 1.0, ...]
// 1 (right): [0.1, 0.9, ...]
// ...
//
// @note: each channel is an `Arc` that can be cloned by the audio nodes, we
// only need Arc and not Mutex here because if a node acquired a channel,
// it must keep a reference to it even if `set_channel_data` is called. In such
// situation we just replace the channel with a new `Arc` pointing to new `Vec`.
// So no mutation occurs and memory allocation takes place in the control thread.
// Having the `Arc` at the level of the single channel, will probably lead to a
// bit more bookkeeping in the node side, but allows to be a bit more efficent in
// terms of memory, as we can discard only one channel at a time.
//
// @note - spec does not define "audio asset" which can lead to confusion
//
// @see - <https://webaudio.github.io/web-audio-api/#acquire-the-content>
#[derive(Clone)]
pub(crate) struct AudioBufferData {
    channels: Rc<Vec<Arc<Vec<f32>>>>
}

impl AudioBufferData {
    // only used by AudioBuffer
    fn new(number_of_channels: usize, length: usize) -> Self {
        let mut channels = Vec::<Arc<Vec<f32>>>::with_capacity(number_of_channels);
        // [spec] Note: This initializes the underlying storage to zero.
        // we don't want to clone just the Arc here, but the whole underlying data,
        // so we use `fill_with` instead of `fill`
        // internal_data.fill_with(|| Arc::new(vec![0.; options.length as usize]));
        for _ in 0..number_of_channels {
            let mut channel = Vec::<f32>::with_capacity(length);
            channel.resize(10, 0.);

            channels.push(Arc::new(channel));
        }

        Self { channels: Rc::new(channels) }
    }
}

// used in `decodeAudioData`
impl From<Vec<Vec<f32>>> for AudioBufferData {
    fn from(decoded: Vec<Vec<f32>>) -> Self {
        let number_of_channels = decoded.len();
        // wrap each decoded channel with `Arc
        let mut channels = Vec::<Arc<Vec<f32>>>::with_capacity(number_of_channels);
        // @note - is there a better way than using `to_vec`?
        // basically, is there any way to grab the `decoded.channel` reference and
        // put it where we want it to be without memory allocation?
        // (sometimes JS feels so simple... :)
        for channel_number in 0..number_of_channels {
            let channel = decoded[channel_number].to_vec();
            channels.push(Arc::new(channel));
        }

        Self { channels: Rc::new(channels) }
    }
}

// @note - what could be default/required values in Rust syntax, is this possible?
#[derive(Copy, Clone, Debug)]
pub struct AudioBufferOptions {
    number_of_channels: usize,  // defaults to 1
    length: usize,              // required
    sample_rate: SampleRate,    // required
}

#[derive(Clone)]
pub struct AudioBuffer {
    number_of_channels: usize,
    length: usize,
    sample_rate: SampleRate,
    internal_data: AudioBufferData,
}

// @todo - possible solution to make the `offlineContext.startRendering` API compliant
// impl From<buffer::AudioBuffer> for AudioBuffer {}

impl AudioBuffer {
    // https://webaudio.github.io/web-audio-api/#AudioBuffer-constructors
    pub fn new(options: AudioBufferOptions) -> Self {
        let AudioBufferOptions {
            number_of_channels,
            length,
            sample_rate,
        } = options;

        let internal_data = AudioBufferData::new(number_of_channels, length);

        Self {
            number_of_channels,
            length,
            sample_rate,
            internal_data,
        }
    }

    pub fn number_of_channels(&self) -> usize {
        self.number_of_channels
    }

    pub fn length(&self) -> usize {
        self.length
    }

    pub fn sample_rate(&self) -> SampleRate {
        self.sample_rate
    }

    pub fn duration(&self) -> f64 {
        (self.length / self.sample_rate.0 as usize) as f64
    }

    // @note - not sure if we can handle default arguments in another way
    pub fn copy_from_channel(&self, destination: &mut Vec<f32>, channel_number: usize) {
        self.copy_from_channel_with_offset(destination, channel_number, 0);
    }

    pub fn copy_from_channel_with_offset(
        &self,
        destination: &mut Vec<f32>,
        channel_number: usize,
        offset: usize,
    ) {
        // [spec] Let buffer be the AudioBuffer with 𝑁𝑏 frames, let 𝑁𝑓 be the number
        // of elements in the destination array, and 𝑘 be the value of bufferOffset.
        // Then the number of frames copied from buffer to destination is max(0,min(𝑁𝑏−𝑘,𝑁𝑓)).
        // If this is less than 𝑁𝑓, then the remaining elements of destination are not modified.
        //
        // we use the `capacity` instead of `len` because destination is a write
        // buffer and we don't care of its current values (cf. `copy_to_channel`)
        let dest_capacity = destination.capacity();
        let max_frame = (self.length - offset).min(dest_capacity).max(0);
        let channel = &self.internal_data.channels[channel_number];

        for index in 0..max_frame {
            if index < destination.len() {
                destination[index] = channel[index + offset];
            } else {
                destination.push(channel[index + offset]);
            }
        }
    }

    // @note - not sure if we can handle default arguments in another way
    pub fn copy_to_channel(&mut self, source: &Vec<f32>, channel_number: usize) {
        self.copy_to_channel_with_offset(source, channel_number, 0);
    }

    pub fn copy_to_channel_with_offset(&mut self, source: &Vec<f32>, channel_number: usize, offset: usize) {
        // [spec] Let buffer be the AudioBuffer with 𝑁𝑏 frames, let 𝑁𝑓 be the number
        // of elements in the source array, and 𝑘 be the value of bufferOffset. Then
        // the number of frames copied from source to the buffer is max(0,min(𝑁𝑏−𝑘,𝑁𝑓)).
        // If this is less than 𝑁𝑓, then the remaining elements of buffer are not modified.
        //
        // we use the `len` instead of `capacity` because source is a read buffer and we
        // want to be sure we don't access some undefined index (cf. `copy_from_channel`)
        let src_len = source.len();
        let max_frame = (self.length - offset).min(src_len).max(0);
        let channel = &self.internal_data.channels[channel_number];
        // we need to copy the underlying channel here because it could have been
        // acquired by some node and be in use.
        // @see - https://webaudio.github.io/web-audio-api/#acquire-the-content
        let mut copy = channel.to_vec();

        for index in 0..max_frame {
            copy[index + offset] = source[index];
        }

        // replace channel with modified copy and new Arc so that next time some
        // node acquire the content it will grab the updated values and the old
        // data will be freed when last ref from audio node is dropped.
        // The nodes who already acquired the Arc to the previous resource
        // are therefore not impacted.
        //
        // @note - maybe this does not handle properly some edge case where:
        //  `buffer is set to the source -> buffer is changed -> source is started`
        //  but for most use-cases this seems to ok
        let channels = Rc::make_mut(&mut self.internal_data.channels);
        channels[channel_number] = Arc::new(copy);
    }

    // [spec] According to the rules described in acquire the content either allow writing
    // into or getting a copy of the bytes stored in [[internal data]] in a new Float32Array
    //
    // @note - that's really not clear and kind of "do whatever you want unless it
    // breaks something...", so just return a copy to make sure nothing can be messed up
    pub fn get_channel_data(&self, channel_number: usize) -> Vec<f32> {
        let channel = &self.internal_data.channels[channel_number];
        channel.to_vec()
    }

    // give acces to internal_data to nodes so that they can acquire the data
    pub(crate) fn get_channel_clone(&self, channel_number: usize) -> Arc<Vec<f32>> {
        self.internal_data.channels[channel_number].clone()
    }

    #[cfg(test)]
    pub(crate) fn get_slice(&self, channel_number: usize, start: usize, end: usize) -> &[f32] {
        &self.internal_data.channels[channel_number][start..end]
    }
}

#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;
    use super::*;

    #[test]
    fn test_constructor() {
        let options = AudioBufferOptions {
            number_of_channels: 1,
            length: 10,
            sample_rate: SampleRate(1),
        };

        let audio_buffer = AudioBuffer::new(options);

        assert_eq!(audio_buffer.number_of_channels(), 1);
        assert_eq!(audio_buffer.length(), 10);
        assert_eq!(audio_buffer.sample_rate().0, 1);
        assert_eq!(audio_buffer.duration(), 10.);
    }

    #[test]
    fn test_copy_from_channel() {
        let options = AudioBufferOptions {
            number_of_channels: 1,
            length: 10,
            sample_rate: SampleRate(1),
        };

        let audio_buffer = AudioBuffer::new(options);

        // same size
        let mut dest = vec![1.; 10];
        audio_buffer.copy_from_channel(&mut dest, 0);
        assert_float_eq!(dest[..], vec![0.; 10][..], abs_all <= 0.);

        // smaller destination
        let mut dest = vec![1.; 5];
        audio_buffer.copy_from_channel(&mut dest, 0);
        assert_float_eq!(
            dest[..],
            [0., 0., 0., 0., 0.][..],
            abs_all <= 0.
        );

        // larger destination
        let mut dest = vec![1.; 11];
        audio_buffer.copy_from_channel(&mut dest, 0);
        assert_float_eq!(
            dest[..],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.][..],
            abs_all <= 0.
        );

        // with offset
        let mut dest = vec![1.; 10];
        audio_buffer.copy_from_channel_with_offset(&mut dest, 0, 5);
        assert_float_eq!(
            dest[..],
            [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.][..],
            abs_all <= 0.
        );
    }

    #[test]
    fn test_copy_to_channel() {
        let options = AudioBufferOptions {
            number_of_channels: 1,
            length: 10,
            sample_rate: SampleRate(1),
        };

        {   // same size
            let mut audio_buffer = AudioBuffer::new(options);
            let mut src = vec![1.; 10];
            audio_buffer.copy_to_channel(&mut src, 0);
            assert_float_eq!(
                audio_buffer.internal_data.channels[0][..],
                [1.; 10][..],
                abs_all <= 0.
            );
        }

        {   // smaller source
            let mut audio_buffer = AudioBuffer::new(options);
            let mut src = vec![1.; 5];
            audio_buffer.copy_to_channel(&mut src, 0);
            assert_float_eq!(
                audio_buffer.internal_data.channels[0][..],
                [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.][..],
                abs_all <= 0.
            );
        }

        {   // larger source
            let mut audio_buffer = AudioBuffer::new(options);
            let mut src = vec![1.; 12];
            audio_buffer.copy_to_channel(&mut src, 0);
            assert_float_eq!(
                audio_buffer.internal_data.channels[0][..],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.][..],
                abs_all <= 0.
            );
        }

        {   // w/ offset
            let mut audio_buffer = AudioBuffer::new(options);
            let mut src = vec![1.; 10];
            audio_buffer.copy_to_channel_with_offset(&mut src, 0, 5);
            assert_float_eq!(
                audio_buffer.internal_data.channels[0][..],
                [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.][..],
                abs_all <= 0.
            );
        }
    }

    #[test]
    fn test_get_channel_data() {
        let options = AudioBufferOptions {
            number_of_channels: 1,
            length: 10,
            sample_rate: SampleRate(1),
        };

        let audio_buffer = AudioBuffer::new(options);
        let mut channel = audio_buffer.get_channel_data(0);
        assert_float_eq!(channel[..], [0.; 10][..], abs_all <= 0.);
        // mutate channel and make sure this does not propagate to internal_data
        channel[0] = 1.;
        assert_float_eq!(
            audio_buffer.internal_data.channels[0][..], [0.; 10][..], abs_all <= 0.
        );
    }

    #[test]
    fn test_is_clonable() {
        let options = AudioBufferOptions {
            number_of_channels: 1,
            length: 10,
            sample_rate: SampleRate(1),
        };

        let audio_buffer = AudioBuffer::new(options);
        let cloned = audio_buffer.clone();

        assert_float_eq!(
            cloned.internal_data.channels[0][..],
            audio_buffer.internal_data.channels[0][..],
            abs_all <= 0.
        );
    }

    #[test]
    fn test_decode_audio_data() {
        let file = std::fs::File::open("sample.wav").unwrap();
        let audio_buffer = decode_audio_data(file);

        println!("----------------------------------------------");
        println!("- number_of_channels: {:?}", audio_buffer.number_of_channels());
        println!("- length: {:?}", audio_buffer.length());
        println!("- sample_rate: {:?}", audio_buffer.sample_rate());
        println!("- duration: {:?}", audio_buffer.duration());

        let left_start = audio_buffer.get_slice(0, 0, 100);
        let right_start = audio_buffer.get_slice(1, 0, 100);

        println!("----------------------------------------------");
        println!("@todo - should check that resampling is ok    ");
        println!("----------------------------------------------");
        println!("- left_start: {:?}", left_start);
        println!("----------------------------------------------");
        println!("- right_start: {:?}", right_start);
    }
}
