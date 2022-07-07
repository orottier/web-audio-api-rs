use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;

use creek::{ReadDiskStream, SeekMode, SymphoniaDecoder};
use crossbeam_channel::{Receiver, Sender};

use crate::{AtomicF64, AudioBuffer, RENDER_QUANTUM_SIZE};

/// Real time safe audio stream
pub(crate) struct RTSStream {
    stream: ReadDiskStream<SymphoniaDecoder>,
    current_time: Arc<AtomicF64>,
    receiver: Receiver<MediaElementAction>,
    paused: bool,
    loop_: bool,
}

/// Controller actions for a media element
pub(crate) enum MediaElementAction {
    /// Seek to the given timestamp
    Seek(f64),
    /// Enable/disable looping
    SetLoop(bool),
    /// Start or restart the stream
    Play,
    /// Pause the stream
    Pause,
}

/// Shim of the `<audio>` element which allows you to efficiently play and seek audio from disk
///
/// The documentation for [`MediaElementAudioSourceNode`](crate::node::MediaElementAudioSourceNode)
/// contains usage instructions.
pub struct MediaElement {
    stream: Option<RTSStream>,
    current_time: Arc<AtomicF64>,
    sender: Sender<MediaElementAction>,
    loop_: bool,
}

impl MediaElement {
    /// Create a new instance for a given file path
    pub fn new<P: Into<PathBuf>>(file: P) -> Result<Self, Box<dyn Error>> {
        // Open a read stream.
        let mut read_disk_stream = ReadDiskStream::<SymphoniaDecoder>::new(
            file,               // Path to file.
            0,                  // The frame in the file to start reading from.
            Default::default(), // Use default read stream options.
        )?;

        // Cache the start of the file into cache with index `0`.
        let _ = read_disk_stream.cache(0, 0);

        // Tell the stream to seek to the beginning of file. This will also alert the stream to the existence
        // of the cache with index `0`.
        read_disk_stream.seek(0, SeekMode::default())?;

        // Wait until the buffer is filled before sending it to the process thread.
        read_disk_stream.block_until_ready()?;

        // Setup control/render thream message bus
        let (sender, receiver) = crossbeam_channel::unbounded();
        // Setup currentTime shared value
        let current_time = Arc::new(AtomicF64::new(0.));

        let rts_stream = RTSStream {
            stream: read_disk_stream,
            current_time: current_time.clone(),
            receiver,
            loop_: false,
            paused: true,
        };

        Ok(Self {
            stream: Some(rts_stream),
            current_time,
            sender,
            loop_: false,
        })
    }

    pub(crate) fn take_stream(&mut self) -> Option<RTSStream> {
        self.stream.take()
    }

    pub fn current_time(&self) -> f64 {
        self.current_time.load()
    }

    pub fn set_current_time(&self, value: f64) {
        let _ = self.sender.send(MediaElementAction::Seek(value));
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
        let sample_rate = self.stream.info().sample_rate.unwrap() as f32;

        if let Ok(msg) = self.receiver.try_recv() {
            match msg {
                MediaElementAction::Seek(value) => {
                    self.current_time.store(value);
                    let frame = (value * sample_rate as f64) as usize;
                    self.stream.seek(frame, SeekMode::default()).unwrap();
                }
                MediaElementAction::SetLoop(value) => {
                    self.loop_ = value;
                }
                MediaElementAction::Play => self.paused = false,
                MediaElementAction::Pause => self.paused = true,
            };
        }

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
                    self.current_time.store(0.);
                } else {
                    let current_time = self.current_time.load();
                    self.current_time
                        .store(current_time + (RENDER_QUANTUM_SIZE as f64 / sample_rate as f64));
                }

                Ok(buf)
            }
            Err(e) => Err(Box::new(e) as _),
        };

        Some(next)
    }
}
