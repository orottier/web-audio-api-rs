use web_audio_api::context::{BaseAudioContext, OfflineAudioContext};

// Use a pool of 2 threads for decoding
const THREAD_POOL_SIZE: usize = 2;

// Decode audio buffers in multiple threads
//
// `cargo run --release --example decode_multithreaded`
pub fn main() {
    env_logger::init();

    // Set up the message channels for job submission and result callback
    let (job_sender, job_receiver) = crossbeam_channel::unbounded();
    let (result_sender, result_receiver) = crossbeam_channel::unbounded();

    // Spawn decoder threads for the thread pool
    for _ in 0..THREAD_POOL_SIZE {
        let job_receiver = job_receiver.clone();
        let result_sender = result_sender.clone();

        std::thread::spawn(move || {
            // Setup an OfflineAudioContext, the actual length and channel count do not matter
            let context = OfflineAudioContext::new(2, 100, 44100.);

            for path in job_receiver.iter() {
                let file = std::fs::File::open(&path).unwrap();
                let audio_buffer = context.decode_audio_data_sync(file);

                // We send back a tuple of the input path and the decoder result
                let result = (path, audio_buffer);
                let _ = result_sender.send(result);
            }
        });
    }

    // Submit decode job for all files in the samples directory
    let job_count = std::fs::read_dir("./samples")
        .unwrap()
        .map(|p| job_sender.send(p.unwrap().path()).unwrap())
        .count();

    // Await the decoded data and print their results
    let mut result_count = 0;
    for result in result_receiver.iter() {
        result_count += 1;
        let path = result.0.display();
        let info = result
            .1
            .map(|buffer| format!("Success - decoded {} samples", buffer.length()))
            .unwrap_or_else(|e| format!("Error - {e:?}"));

        println!("{path} - {info}");

        // We are done when all jobs are handled
        if job_count == result_count {
            break;
        }
    }
}
