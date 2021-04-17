use realfft::RealFftPlanner;

fn void() {
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
