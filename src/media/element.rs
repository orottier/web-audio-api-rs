use std::error::Error;
use std::path::PathBuf;

use creek::{ReadDiskStream, SeekMode, SymphoniaDecoder};

use crate::{AudioBuffer, RENDER_QUANTUM_SIZE};

pub(crate) struct RTSStream(ReadDiskStream<SymphoniaDecoder>);

pub struct MediaElement {
    stream: Option<RTSStream>,
}

impl MediaElement {
    #[allow(clippy::missing_panics_doc)]
    pub fn new<P: Into<PathBuf>>(file: P) -> Self {
        // Open a read stream.
        let mut read_disk_stream = ReadDiskStream::<SymphoniaDecoder>::new(
            file,               // Path to file.
            0,                  // The frame in the file to start reading from.
            Default::default(), // Use default read stream options.
        )
        .unwrap();

        // Cache the start of the file into cache with index `0`.
        let _ = read_disk_stream.cache(0, 0);

        // Tell the stream to seek to the beginning of file. This will also alert the stream to the existence
        // of the cache with index `0`.
        read_disk_stream.seek(0, SeekMode::default()).unwrap();

        // Wait until the buffer is filled before sending it to the process thread.
        //
        // NOTE: Do ***not*** use this method in a real-time thread.
        read_disk_stream.block_until_ready().unwrap();

        Self {
            stream: Some(RTSStream(read_disk_stream)),
        }
    }

    pub(crate) fn take_stream(&mut self) -> Option<RTSStream> {
        self.stream.take()
    }
}

impl Iterator for RTSStream {
    type Item = Result<AudioBuffer, Box<dyn Error + Send + Sync>>;

    fn next(&mut self) -> Option<Self::Item> {
        let sample_rate = self.0.info().sample_rate.unwrap() as f32;

        let next = self
            .0
            .read(RENDER_QUANTUM_SIZE)
            .map(|data| AudioBuffer::from(vec![data.read_channel(0).to_vec()], sample_rate))
            .map_err(|e| Box::new(e) as _);

        Some(next)
    }
}
