use std::error::Error;
use std::path::PathBuf;

use creek::{ReadDiskStream, SeekMode, SymphoniaDecoder};
use crossbeam_channel::{Receiver, Sender};

use crate::{AudioBuffer, RENDER_QUANTUM_SIZE};

pub(crate) struct RTSStream {
    stream: ReadDiskStream<SymphoniaDecoder>,
    receiver: Receiver<MediaElementAction>,
    paused: bool,
    loop_: bool,
}

pub(crate) enum MediaElementAction {
    Seek(usize),
    SetLoop(bool),
    Play,
    Pause,
}

pub struct MediaElement {
    stream: Option<RTSStream>,
    sender: Sender<MediaElementAction>,
    loop_: bool,
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
            loop_: false,
            paused: true,
        };

        Self {
            stream: Some(rts_stream),
            sender,
            loop_: false,
        }
    }

    pub(crate) fn take_stream(&mut self) -> Option<RTSStream> {
        self.stream.take()
    }

    pub fn seek(&self, frame: usize) {
        let _ = self.sender.send(MediaElementAction::Seek(frame));
    }

    pub fn loop_(&self) -> bool {
        self.loop_
    }

    pub fn set_loop(&mut self, value: bool) {
        self.loop_ = value;
        let _ = self.sender.send(MediaElementAction::SetLoop(value));
    }

    pub fn play(&self) {
        let _ = self.sender.send(MediaElementAction::Play);
    }

    pub fn pause(&self) {
        let _ = self.sender.send(MediaElementAction::Pause);
    }
}

impl Iterator for RTSStream {
    type Item = Result<AudioBuffer, Box<dyn Error + Send + Sync>>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Ok(msg) = self.receiver.try_recv() {
            match msg {
                MediaElementAction::Seek(frame) => {
                    self.stream.seek(frame, SeekMode::default()).unwrap();
                }
                MediaElementAction::SetLoop(value) => {
                    self.loop_ = value;
                }
                MediaElementAction::Play => self.paused = false,
                MediaElementAction::Pause => self.paused = true,
            };
        }

        let sample_rate = self.stream.info().sample_rate.unwrap() as f32;

        if self.paused {
            let silence = AudioBuffer::from(vec![vec![0.; RENDER_QUANTUM_SIZE]], sample_rate);
            return Some(Ok(silence));
        }

        let next = match self.stream.read(RENDER_QUANTUM_SIZE) {
            Ok(data) => {
                let channels: Vec<_> = (0..data.num_channels())
                    .map(|i| data.read_channel(i).to_vec())
                    .collect();
                let buf = AudioBuffer::from(channels, sample_rate);

                if self.loop_ && data.reached_end_of_file() {
                    self.stream.seek(0, SeekMode::default()).unwrap();
                }

                Ok(buf)
            }
            Err(e) => Err(Box::new(e) as _),
        };

        Some(next)
    }
}
