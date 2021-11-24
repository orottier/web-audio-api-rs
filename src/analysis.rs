use crate::alloc::ChannelData;
use crate::BUFFER_SIZE;

use realfft::{num_complex::Complex, RealFftPlanner};

use std::f32::consts::PI;

const MAX_QUANTA: usize = 256;
const MAX_SAMPLES: usize = MAX_QUANTA * BUFFER_SIZE;

/// Blackman window values iterator with alpha = 0.16
pub fn generate_blackman(size: usize) -> impl Iterator<Item = f32> {
    let alpha = 0.16;
    let a0 = (1. - alpha) / 2.;
    let a1 = 1. / 2.;
    let a2 = alpha / 2.;

    (0..size).map(move |i| {
        a0 - a1 * (2. * PI * i as f32 / size as f32).cos()
            + a2 * (4. * PI * i as f32 / size as f32).cos()
    })
}

/// Ring buffer for time domain analysis
struct TimeAnalyser {
    buffer: Vec<ChannelData>,
    index: u8,
    previous_cycle_index: u8,
}

impl TimeAnalyser {
    /// Create a new TimeAnalyser
    fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(MAX_QUANTA),
            index: 0,
            previous_cycle_index: 0,
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

    /// Check if we have completed a full round of `fft_size` samples
    fn check_complete_cycle(&mut self, fft_size: usize) -> bool {
        // number of buffers processed since last complete cycle
        let processed = self.index.wrapping_sub(self.previous_cycle_index);
        let processed_samples = processed as usize * BUFFER_SIZE;

        // cycle is complete when divisible by fft_size
        if processed_samples % fft_size == 0 {
            self.previous_cycle_index = self.index;
            return true;
        }

        false
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
        let buf_chunks = buffer[0..true_size].chunks_mut(BUFFER_SIZE).rev();

        // copy data from internal buffer to output buffer
        buf_chunks
            .zip(data_chunks)
            .for_each(|(b, d)| b.copy_from_slice(&d[..b.len()]));
    }
}

/// Analyser kernel for time domain and frequency data
pub(crate) struct Analyser {
    time: TimeAnalyser,

    fft_planner: RealFftPlanner<f32>,
    fft_input: Vec<f32>,
    fft_scratch: Vec<Complex<f32>>,
    fft_output: Vec<Complex<f32>>,

    current_fft_size: usize,
    previous_block: Vec<f32>,
    blackman: Vec<f32>,
}

impl Analyser {
    /// Create a new analyser kernel
    pub fn new(initial_fft_size: usize) -> Self {
        let mut fft_planner = RealFftPlanner::<f32>::new();
        let max_fft = fft_planner.plan_fft_forward(MAX_SAMPLES);

        let fft_input = max_fft.make_input_vec();
        let fft_scratch = max_fft.make_scratch_vec();
        let fft_output = max_fft.make_output_vec();
        let previous_block = vec![0.; fft_output.len()];

        // precalculate Blackman window values, reserve enough space for all input sizes
        let mut blackman = Vec::with_capacity(fft_input.len());
        generate_blackman(initial_fft_size).for_each(|v| blackman.push(v));

        Self {
            time: TimeAnalyser::new(),
            fft_planner,
            fft_input,
            fft_scratch,
            fft_output,
            current_fft_size: initial_fft_size,
            previous_block,
            blackman,
        }
    }

    pub fn current_fft_size(&self) -> usize {
        self.current_fft_size
    }

    /// Add samples to the ring buffer
    pub fn add_data(&mut self, data: ChannelData) {
        self.time.add_data(data);
    }

    /// Read out the time domain ring buffer (max `fft_size samples)
    pub fn get_float_time(&self, buffer: &mut [f32], fft_size: usize) {
        self.time.get_float_time(buffer, fft_size);
    }

    /// Check if we have completed a full round of `fft_size` samples
    pub fn check_complete_cycle(&mut self, fft_size: usize) -> bool {
        self.time.check_complete_cycle(fft_size)
    }

    /// Copy the frequency data
    pub fn get_float_frequency(&mut self, buffer: &mut [f32]) {
        let previous_block = &mut self.previous_block[..self.current_fft_size / 2 + 1];

        // nomalizing, conversion to dB and fill buffer
        let norm = 20. * (self.current_fft_size as f32).sqrt().log10();
        buffer
            .iter_mut()
            .zip(previous_block.iter())
            .for_each(|(b, o)| *b = 20. * o.log10() - norm);
    }

    /// Calculate the frequency data
    pub fn calculate_float_frequency(&mut self, fft_size: usize, smoothing_time_constant: f32) {
        // reset state after resizing
        if self.current_fft_size != fft_size {
            // previous block data
            self.previous_block[0..fft_size / 2 + 1]
                .iter_mut()
                .for_each(|v| *v = 0.);

            // blackman window
            self.blackman.clear();
            generate_blackman(fft_size).for_each(|v| self.blackman.push(v));

            self.current_fft_size = fft_size;
        }

        let r2c = self.fft_planner.plan_fft_forward(fft_size);

        // setup proper sized buffers
        let input = &mut self.fft_input[..fft_size];
        let output = &mut self.fft_output[..fft_size / 2 + 1];
        let scratch = &mut self.fft_scratch[..r2c.get_scratch_len()];
        let previous_block = &mut self.previous_block[..fft_size / 2 + 1];

        // put time domain data in fft_input
        self.time.get_float_time(input, fft_size);

        // blackman window
        input
            .iter_mut()
            .zip(self.blackman.iter())
            .for_each(|(i, b)| *i *= *b);

        // calculate frequency data
        r2c.process_with_scratch(input, output, scratch).unwrap();

        // smoothing over time
        previous_block
            .iter_mut()
            .zip(output.iter())
            .for_each(|(p, c)| {
                *p = smoothing_time_constant * *p + (1. - smoothing_time_constant) * c.norm()
            });
    }
}

#[cfg(test)]
mod tests {
    use float_eq::{assert_float_eq, float_eq};

    use super::*;

    use crate::alloc::Alloc;

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
        let mut buffer = vec![-1.; BUFFER_SIZE * 5];

        // feed single data buffer
        analyser.add_data(alloc.silence());

        // get data, should be padded with zeroes
        analyser.get_float_time(&mut buffer[..], BUFFER_SIZE * 5);
        assert_float_eq!(&buffer[..], &[0.; 5 * BUFFER_SIZE][..], abs_all <= 0.);

        // feed data for more than 256 times (the ring buffer size)
        for i in 0..258 {
            let mut signal = alloc.silence();
            // signal = i
            signal.copy_from_slice(&[i as f32; BUFFER_SIZE]);
            analyser.add_data(signal);
        }

        // this should return non-zero data now
        analyser.get_float_time(&mut buffer[..], BUFFER_SIZE * 4);

        // taken from the end of the ring buffer
        assert_float_eq!(
            &buffer[0..BUFFER_SIZE],
            &[254.; BUFFER_SIZE][..],
            abs_all <= 0.
        );
        assert_float_eq!(
            &buffer[BUFFER_SIZE..2 * BUFFER_SIZE],
            &[255.; BUFFER_SIZE][..],
            abs_all <= 0.
        );
        // taken from the start of the ring buffer
        assert_float_eq!(
            &buffer[2 * BUFFER_SIZE..3 * BUFFER_SIZE],
            &[256.; BUFFER_SIZE][..],
            abs_all <= 0.
        );
        assert_float_eq!(
            &buffer[3 * BUFFER_SIZE..4 * BUFFER_SIZE],
            &[257.; BUFFER_SIZE][..],
            abs_all <= 0.
        );
        // excess capacity should be left unaltered
        assert_float_eq!(
            &buffer[4 * BUFFER_SIZE..5 * BUFFER_SIZE],
            &[0.; BUFFER_SIZE][..],
            abs_all <= 0.
        );

        // check for small fft_size
        buffer.resize(32, 0.);
        analyser.get_float_time(&mut buffer[..], BUFFER_SIZE);
        assert_float_eq!(&buffer[..], &[257.; 32][..], abs_all <= 0.);
    }

    #[test]
    fn test_complete_cycle() {
        let alloc = Alloc::with_capacity(256);
        let mut analyser = TimeAnalyser::new();

        // check values smaller than BUFFER_SIZE
        analyser.add_data(alloc.silence());
        assert!(analyser.check_complete_cycle(32));

        // check BUFFER_SIZE
        analyser.add_data(alloc.silence());
        assert!(analyser.check_complete_cycle(BUFFER_SIZE));

        // check multiple of BUFFER_SIZE
        analyser.add_data(alloc.silence());
        assert!(!analyser.check_complete_cycle(BUFFER_SIZE * 2));
        analyser.add_data(alloc.silence());
        assert!(analyser.check_complete_cycle(BUFFER_SIZE * 2));
        analyser.add_data(alloc.silence());
        assert!(!analyser.check_complete_cycle(BUFFER_SIZE * 2));
    }

    #[test]
    fn test_freq_domain() {
        let alloc = Alloc::with_capacity(256);

        let fft_size: usize = BUFFER_SIZE * 4;
        let mut analyser = Analyser::new(fft_size);
        let mut buffer = vec![-1.; fft_size];

        // feed single data buffer
        analyser.add_data(alloc.silence());

        // get data, should be zero (negative infinity decibel)
        analyser.calculate_float_frequency(fft_size, 0.8);
        analyser.get_float_frequency(&mut buffer[..]);

        // only N / 2 + 1 values should contain frequency data, rest is unaltered
        assert!(buffer[0..BUFFER_SIZE * 2 + 1] == [f32::NEG_INFINITY; BUFFER_SIZE * 2 + 1]);
        assert_float_eq!(
            &buffer[2 * BUFFER_SIZE + 1..],
            &[-1.; 2 * BUFFER_SIZE - 1][..],
            abs_all <= 0.
        );

        // feed data for more than 256 times (the ring buffer size)
        for i in 0..258 {
            let mut signal = alloc.silence();
            // signal = i
            signal.copy_from_slice(&[i as f32; BUFFER_SIZE]);
            analyser.add_data(signal);
        }

        // this should return other data now
        analyser.calculate_float_frequency(fft_size, 0.8);
        analyser.get_float_frequency(&mut buffer[..]);
        assert!(buffer[0..BUFFER_SIZE * 2 + 1] != [f32::NEG_INFINITY; BUFFER_SIZE * 2 + 1]);
    }

    #[test]
    fn test_blackman() {
        let values: Vec<f32> = generate_blackman(2048).collect();

        let min = values
            .iter()
            .fold(1000., |min, &val| if val < min { val } else { min });
        let max = values
            .iter()
            .fold(0., |max, &val| if val > max { val } else { max });
        assert!(min < 0.01 && min > 0.);
        assert!(max > 0.99 && max <= 1.);

        let min_pos = values
            .iter()
            .position(|&v| float_eq!(v, min, abs_all <= 0.))
            .unwrap();
        let max_pos = values
            .iter()
            .position(|&v| float_eq!(v, max, abs_all <= 0.))
            .unwrap();
        assert_eq!(min_pos, 0);
        assert_eq!(max_pos, 1024);
    }
}
