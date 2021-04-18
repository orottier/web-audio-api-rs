use crate::alloc::ChannelData;

use realfft::RealFftPlanner;

fn fft() {
    let length = 256;

    // make a planner
    let mut real_planner = RealFftPlanner::<f64>::new();

    // create a FFT
    let r2c = real_planner.plan_fft_forward(length);
    // make input and output vectors
    let mut indata = r2c.make_input_vec();
    let mut spectrum = r2c.make_output_vec();

    // Are they the length we expect?
    assert_eq!(indata.len(), length);
    assert_eq!(spectrum.len(), length / 2 + 1);

    // Forward transform the input data
    r2c.process(&mut indata, &mut spectrum).unwrap();

    // create an iFFT and an output vector
    let c2r = real_planner.plan_fft_inverse(length);
    let mut outdata = c2r.make_output_vec();
    assert_eq!(outdata.len(), length);

    c2r.process(&mut spectrum, &mut outdata).unwrap();
}

pub(crate) struct Analyser {
    buffer: Vec<ChannelData>,
    index: u8,
}

impl Analyser {
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(256),
            index: 0,
        }
    }

    pub fn add_data(&mut self, data: ChannelData) {
        if self.buffer.len() < 256 {
            self.buffer.push(data);
        } else {
            self.buffer[self.index as usize] = data;
        }
        self.index = self.index.wrapping_add(1);
    }

    pub fn get_float_time(&self, buffer: &mut [f32], fft_size: usize) {
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
        let buf_chunks = buffer[0..true_size]
            .chunks_mut(crate::BUFFER_SIZE as usize)
            .rev();

        // copy data from internal buffer to output buffer
        buf_chunks
            .zip(data_chunks)
            .for_each(|(b, d)| b.copy_from_slice(&d[..b.len()]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::alloc::Alloc;
    const LEN: usize = crate::BUFFER_SIZE as usize;

    #[test]
    fn test_time_domain() {
        let alloc = Alloc::with_capacity(256);

        let mut analyser = Analyser::new();
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
}
