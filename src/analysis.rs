use crate::alloc::ChannelData;
use crate::BUFFER_SIZE;

use realfft::{num_complex::Complex, RealFftPlanner};

const MAX_QUANTA: usize = 256;
const MAX_SAMPLES: usize = MAX_QUANTA * BUFFER_SIZE as usize;

/// Ring buffer for time domain analysis
struct TimeAnalyser {
    buffer: Vec<ChannelData>,
    index: u8,
}

impl TimeAnalyser {
    /// Create a new TimeAnalyser
    fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(MAX_QUANTA),
            index: 0,
        }
    }

    /// Add samples to the ring buffer
    fn add_data(&mut self, data: ChannelData) {
        if self.buffer.len() < 256 {
            self.buffer.push(data);
        } else {
            self.buffer[self.index as usize] = data;
        }
        self.index = self.index.wrapping_add(1);
    }

    /// Read out the ring buffer (max `fft_size` samples)
    fn get_float_time(&self, buffer: &mut [f32], fft_size: usize) {
        // buffer is never empty when this call is made
        debug_assert!(!self.buffer.is_empty());

        // get a reference to the 'silence buffer'
        let silence = self.buffer[0].silence();

        // order the ring buffer, and pad with silence
        let data_chunks = self.buffer[self.index as usize..]
            .iter()
            .chain(self.buffer[..self.index as usize].iter())
            .rev()
            .chain(std::iter::repeat(&silence));

        // split the output buffer in same sized chunks
        let true_size = fft_size.min(buffer.len());
        let buf_chunks = buffer[0..true_size].chunks_mut(BUFFER_SIZE as usize).rev();

        // copy data from internal buffer to output buffer
        buf_chunks
            .zip(data_chunks)
            .for_each(|(b, d)| b.copy_from_slice(&d[..b.len()]));
    }
}

/// Analyser kerner for time domain and frequency data
pub(crate) struct Analyser {
    time: TimeAnalyser,

    fft_planner: RealFftPlanner<f32>,
    fft_input: Vec<f32>,
    fft_scratch: Vec<Complex<f32>>,
    fft_output: Vec<Complex<f32>>,
}

impl Analyser {
    /// Create a new analyser kernel with max capacity
    pub fn new() -> Self {
        let mut fft_planner = RealFftPlanner::<f32>::new();
        let max_fft = fft_planner.plan_fft_forward(MAX_SAMPLES);

        Self {
            time: TimeAnalyser::new(),
            fft_planner,
            fft_input: max_fft.make_input_vec(),
            fft_scratch: max_fft.make_scratch_vec(),
            fft_output: max_fft.make_output_vec(),
        }
    }

    /// Add samples to the ring buffer
    pub fn add_data(&mut self, data: ChannelData) {
        self.time.add_data(data);
    }

    /// Read out the time domain ring buffer (max `fft_size samples)
    pub fn get_float_time(&self, buffer: &mut [f32], fft_size: usize) {
        self.time.get_float_time(buffer, fft_size);
    }

    /// Calculate the frequency data
    pub fn get_float_frequency(&mut self, buffer: &mut [f32], fft_size: usize) {
        let r2c = self.fft_planner.plan_fft_forward(fft_size);

        // setup proper sized buffers
        let input = &mut self.fft_input[..fft_size];
        let output = &mut self.fft_output[..fft_size / 2 + 1];
        let scratch = &mut self.fft_scratch[..r2c.get_scratch_len()];

        // put time domain data in fft_input
        self.time.get_float_time(input, fft_size);

        // blackman window

        // calculate frequency data
        r2c.process_with_scratch(input, output, scratch).unwrap();

        // smoothing

        // nomalizing, conversion to dB and fill buffer
        let norm = 20. * (fft_size as f32).sqrt().log10();
        buffer
            .iter_mut()
            .zip(output.iter())
            .for_each(|(b, o)| *b = 20. * o.norm().log10() - norm);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::alloc::Alloc;
    const LEN: usize = BUFFER_SIZE as usize;

    #[test]
    fn assert_index_size() {
        // silly test to remind us MAX_QUANTA should wrap around a u8,
        // otherwise the ring buffer index breaks
        assert_eq!(u8::MAX as usize + 1, MAX_QUANTA);
    }

    #[test]
    fn test_time_domain() {
        let alloc = Alloc::with_capacity(256);

        let mut analyser = TimeAnalyser::new();
        let mut buffer = vec![-1.; LEN * 5];

        // feed single data buffer
        analyser.add_data(alloc.silence());

        // get data, should be padded with zeroes
        analyser.get_float_time(&mut buffer[..], LEN * 5);
        assert_eq!(&buffer[..], &[0.; LEN * 5]);

        // feed data for more than 256 times (the ring buffer size)
        for i in 0..258 {
            let mut signal = alloc.silence();
            // signal = i
            signal.copy_from_slice(&[i as f32; LEN]);
            analyser.add_data(signal);
        }

        // this should return non-zero data now
        analyser.get_float_time(&mut buffer[..], LEN * 4);

        // taken from the end of the ring buffer
        assert_eq!(&buffer[0..LEN], &[254.; LEN]);
        assert_eq!(&buffer[LEN..LEN * 2], &[255.; LEN]);

        // taken from the start of the ring buffer
        assert_eq!(&buffer[LEN * 2..LEN * 3], &[256.; LEN]);
        assert_eq!(&buffer[LEN * 3..LEN * 4], &[257.; LEN]);

        // excess capacity should be left unaltered
        assert_eq!(&buffer[LEN * 4..LEN * 5], &[0.; LEN]);

        // check for small fft_size
        buffer.resize(32, 0.);
        analyser.get_float_time(&mut buffer[..], LEN);
        assert_eq!(&buffer[..], &[257.; 32]);
    }

    #[test]
    fn test_freq_domain() {
        let alloc = Alloc::with_capacity(256);
        let mut analyser = Analyser::new();
        let mut buffer = vec![-1.; LEN * 4];

        // feed single data buffer
        analyser.add_data(alloc.silence());

        // get data, should be zero (negative infinity decibel)
        analyser.get_float_frequency(&mut buffer[..], LEN * 4);
        assert_eq!(&buffer[0..LEN * 2 + 1], &[f32::NEG_INFINITY; LEN * 2 + 1]);
        // only N / 2 + 1 values should contain frequency data, rest is unaltered
        assert_eq!(&buffer[LEN * 2 + 1..], &[-1.; LEN * 2 - 1]);

        // feed data for more than 256 times (the ring buffer size)
        for i in 0..258 {
            let mut signal = alloc.silence();
            // signal = i
            signal.copy_from_slice(&[i as f32; LEN]);
            analyser.add_data(signal);
        }

        // this should return other data now
        analyser.get_float_frequency(&mut buffer[..], LEN * 4);
        assert!(&buffer[0..LEN * 2 + 1] != &[f32::NEG_INFINITY; LEN * 2 + 1]);
    }
}
