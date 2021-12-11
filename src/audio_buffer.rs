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
    let number_of_channels = channels as usize;
    let length = reader.duration() as usize; // this is the number of samples per channel

    // println!("channels {:?}", channels);
    // println!("length {:?}", length);
    // println!("sample_rate {:?}", sample_rate);
    // println!("bits_per_sample {:?}", bits_per_sample);
    // println!("sample_format {:?}", sample_format);

    // @note - we don't want the Arc now because we need to have mut access
    // and we don't want to go into Mutex as nthing is mutable after this step
    let mut decoded: Vec::<Vec<f32>> = Vec::with_capacity(number_of_channels);
    // init each channel with empty Vec
    for _ in 0..number_of_channels {
        decoded.push(Vec::<f32>::with_capacity(length));
    }

    match sample_format {
        hound::SampleFormat::Int => {
            // channel are interleaved, so we need to de-interleave
            let mut channel_number = 0;

            if bits_per_sample == 16 {
                for sample in reader.samples::<i16>() {
                    let s = sample.unwrap() as f32 / i16::MAX as f32;
                    decoded[channel_number].push(s);
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
                channel_number = (channel_number + 1) % number_of_channels;
            }
        }
    }

    assert_eq!(decoded[0].len(), length); // duration is ok
    println!("decoded length: {} - input length: {}", decoded[0].len(), length);
    // now we should resample

    // wrap decoded channels into Arc to create the `AudioBuffer`
    // @todo - can (very) probably be done in a much more efficient way...
    let mut internal_data = Vec::<Arc<Vec<f32>>>::with_capacity(number_of_channels);

    for channel_number in 0..number_of_channels {
        let mut channel = Vec::<f32>::with_capacity(length);
        channel.extend_from_slice(&decoded[channel_number][..]);
        internal_data.push(Arc::new(channel));
    }

    AudioBuffer {
        number_of_channels,
        length,
        // @todo - this is wrong, should be the AudioContext.sampleRate
        sample_rate: SampleRate(sample_rate),
        internal_data
    }
}

// @note - what could be default in Rust syntax? is this possible?
// is this even possible to have only partial defaults?
// not really a big deal though
#[derive(Copy, Clone, Debug)]
pub struct AudioBufferOptions {
    number_of_channels: usize, // default to 1
    length: usize,             // required
    sample_rate: SampleRate, // required
}

pub struct AudioBuffer {
    number_of_channels: usize,
    length: usize,
    sample_rate: SampleRate,

    // `internal_data` is a matrix representing a de-interleaved in memory audio asset
    // 0 (left) : [0.0, 1.0, ...]
    // 1 (right): [0.1, 0.9, ...]
    // ...
    //
    // @note: each channel is an `Arc` that can be acquired by the audio nodes,
    // important - the inner `Arc<Vec>`s should be mutated, only replaced by another
    // @note - probably there is possible improvements in term of architecture
    //  and type system
    //
    // @see - <https://webaudio.github.io/web-audio-api/#acquire-the-content>
    //
    // @note - we could maybe reuse buffer::ChannelData here, but that might be also
    // a bit over-engineered as we don't need any sort of memory management strategy
    // (at least as a first step)
    internal_data: Vec<Arc<Vec<f32>>>,
}


impl AudioBuffer {
    // https://webaudio.github.io/web-audio-api/#AudioBuffer-constructors
    pub fn new(options: AudioBufferOptions) -> Self {
        let number_of_channels = options.number_of_channels;
        let length = options.length;

        let mut channels = Vec::<Arc<Vec<f32>>>::with_capacity(number_of_channels);
        // [spec] Note: This initializes the underlying storage to zero.
        // we don't want to clone just the Arc here, but the whole underlying data,
        // so we use `fill_with` instead of `fill`
        // internal_data.fill_with(|| Arc::new(vec![0.; options.length as usize]));
        for _ in 0..number_of_channels {
            let channel = Arc::new(vec![0.; length]);
            channels.push(channel);
        }

        Self {
            number_of_channels,
            length,
            sample_rate: options.sample_rate,
            internal_data: channels,
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
        (self.length * self.sample_rate.0 as usize) as f64
    }

    // @todo - check that usage of `capacity` and `len` are ok
    pub fn copy_from_channel(&self, destination: &mut Vec<f32>, channel_number: usize) {
        self.copy_from_channel_with_offset(destination, channel_number, 0);
    }

    pub fn copy_from_channel_with_offset(
        &self,
        destination: &mut Vec<f32>,
        channel_number: usize,
        offset: usize,
    ) {
        // Let buffer be the AudioBuffer with 𝑁𝑏 frames, let 𝑁𝑓 be the number of
        // elements in the destination array, and 𝑘 be the value of bufferOffset.
        // Then the number of frames copied from buffer to destination is max(0,min(𝑁𝑏−𝑘,𝑁𝑓)).
        // If this is less than 𝑁𝑓, then the remaining elements of destination are not modified.
        let dest_capacity = destination.capacity();
        let max_frame = (self.length - offset).min(dest_capacity).max(0);
        let channel = &self.internal_data[channel_number];

        for index in 0..max_frame {
            if index < destination.len() {
                destination[index] = channel[index + offset];
            } else {
                destination.push(channel[index + offset]);
            }
        }
    }

    pub fn copy_to_channel(&mut self, source: &Vec<f32>, channel_number: usize) {
        self.copy_to_channel_with_offset(source, channel_number, 0);
    }

    // some pub(crate) variation of this one without the memory allocation
    // should be implemented for use by `audioContext.decodeAudioData(binary)`
    pub fn copy_to_channel_with_offset(&mut self, source: &Vec<f32>, channel_number: usize, offset: usize) {
        // Let buffer be the AudioBuffer with 𝑁𝑏 frames, let 𝑁𝑓 be the number of
        // elements in the source array, and 𝑘 be the value of bufferOffset. Then
        // the number of frames copied from source to the buffer is max(0,min(𝑁𝑏−𝑘,𝑁𝑓)).
        // If this is less than 𝑁𝑓, then the remaining elements of buffer are not modified.
        //
        // we use the `len` instead of the capacity to make sur we don't try to
        // access some undefined index
        let src_len = source.len();
        let max_frame = (self.length - offset).min(src_len).max(0);
        let channel = &self.internal_data[channel_number];
        // we need to copy the underlying channel here because it could have been
        // acquired by some node and be in use.
        // @see - https://webaudio.github.io/web-audio-api/#acquire-the-content
        let mut copy = channel.to_vec();

        for index in 0..max_frame {
            copy[index + offset] = source[index];
        }

        // replace channel with modified copy and new Arc so that next time some
        // node acquire the content it will grab the updated values and the old
        // data will be freed when last ref from node will be dropped.
        // The nodes who already acquired the Arc to the previous resource
        // are therefore not impacted.
        self.internal_data[channel_number] = Arc::new(copy);
    }

    // According to the rules described in acquire the content either allow writing
    // into or getting a copy of the bytes stored in [[internal data]] in a new Float32Array
    //
    // that's not really clear, just return a copy to make sure nothing can be messed up
    pub fn get_channel_data(&self, channel_number: usize) -> Vec<f32> {
        let channel = &self.internal_data[channel_number];
        channel.to_vec()
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
            // println!("> {:?}", audio_buffer.internal_data[0]);
            assert_float_eq!(
                audio_buffer.internal_data[0][..],
                [1.; 10][..],
                abs_all <= 0.
            );
        }

        {   // smaller source
            let mut audio_buffer = AudioBuffer::new(options);
            let mut src = vec![1.; 5];
            audio_buffer.copy_to_channel(&mut src, 0);
            // println!("> {:?}", audio_buffer.internal_data[0]);
            assert_float_eq!(
                audio_buffer.internal_data[0][..],
                [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.][..],
                abs_all <= 0.
            );
        }

        {   // larger source
            let mut audio_buffer = AudioBuffer::new(options);
            let mut src = vec![1.; 12];
            audio_buffer.copy_to_channel(&mut src, 0);
            // println!("> {:?}", audio_buffer.internal_data[0]);
            assert_float_eq!(
                audio_buffer.internal_data[0][..],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.][..],
                abs_all <= 0.
            );
        }

        {   // w/ offset
            let mut audio_buffer = AudioBuffer::new(options);
            let mut src = vec![1.; 10];
            audio_buffer.copy_to_channel_with_offset(&mut src, 0, 5);
            // println!("> {:?}", audio_buffer.internal_data[0]);
            assert_float_eq!(
                audio_buffer.internal_data[0][..],
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
            audio_buffer.internal_data[0][..], [0.; 10][..], abs_all <= 0.
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

        let mut left_start = vec![];
        let mut right_start = vec![];
        left_start.extend_from_slice(&audio_buffer.internal_data[0][0..100]);
        right_start.extend_from_slice(&audio_buffer.internal_data[1][0..100]);

        println!("----------------------------------------------");
        println!("- left_start: {:?}", left_start);
        println!("----------------------------------------------");
        println!("- right_start: {:?}", right_start);
    }
}
