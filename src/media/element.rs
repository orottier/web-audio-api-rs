use std::error::Error;
use std::path::PathBuf;

use creek::{ReadDiskStream, SeekMode, SymphoniaDecoder};
use crossbeam_channel::{Receiver, Sender};

use crate::{AudioBuffer, RENDER_QUANTUM_SIZE};

pub(crate) struct RTSStream {
    stream: ReadDiskStream<SymphoniaDecoder>,
    receiver: Receiver<MediaElementAction>,
}

pub(crate) enum MediaElementAction {
    Seek(usize),
    // Pause,
    // Loop,
}

pub struct MediaElement {
    stream: Option<RTSStream>,
    sender: Sender<MediaElementAction>,
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

        // Setup control/render thream message bus
        let (sender, receiver) = crossbeam_channel::unbounded();

        let rts_stream = RTSStream {
            stream: read_disk_stream,
            receiver,
        };

        Self {
            stream: Some(rts_stream),
            sender,
        }
    }

    pub(crate) fn take_stream(&mut self) -> Option<RTSStream> {
        self.stream.take()
    }

    pub fn seek(&self, frame: usize) {
        let _ = self.sender.send(MediaElementAction::Seek(frame));
    }
}

impl Iterator for RTSStream {
    type Item = Result<AudioBuffer, Box<dyn Error + Send + Sync>>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Ok(msg) = self.receiver.try_recv() {
            match msg {
                MediaElementAction::Seek(frame) => {
                    self.stream.seek(frame, SeekMode::default()).unwrap()
                }
            };
        }

        let sample_rate = self.stream.info().sample_rate.unwrap() as f32;

        let next = match self.stream.read(RENDER_QUANTUM_SIZE) {
            Ok(data) => {
                let channels: Vec<_> = (0..data.num_channels())
                    .map(|i| data.read_channel(i).to_vec())
                    .collect();
                let buf = AudioBuffer::from(channels, sample_rate);
                Ok(buf)
            },
            Err(e) => Err(Box::new(e) as _),
        };

        Some(next)
    }
}
