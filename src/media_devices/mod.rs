//! Primitives of the MediaDevices API
//!
//! The MediaDevices interface provides access to connected media input devices like microphones.
//!
//! <https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices>

use crate::context::AudioContextOptions;
use crate::media_streams::MediaStream;

/// List the available media output devices, such as speakers, headsets, loopbacks, etc
///
/// The media device_id can be used to specify the [`sink_id` of the `AudioContext`](crate::context::AudioContextOptions::sink_id)
///
/// ```no_run
/// use web_audio_api::media_devices::{enumerate_devices, MediaDeviceInfoKind};
///
/// let devices = enumerate_devices();
/// assert_eq!(devices[0].device_id(), "1");
/// assert_eq!(devices[0].group_id(), None);
/// assert_eq!(devices[0].kind(), MediaDeviceInfoKind::AudioOutput);
/// assert_eq!(devices[0].label(), "Macbook Pro Builtin Speakers");
/// ```
pub fn enumerate_devices() -> Vec<MediaDeviceInfo> {
    crate::io::enumerate_devices()
}

/// Describes input/output type of a media device
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MediaDeviceInfoKind {
    VideoInput,
    AudioInput,
    AudioOutput,
}

/// Describes a single media input or output device
///
/// Call [`enumerate_devices`] to obtain a list of devices for your hardware.
#[derive(Debug)]
pub struct MediaDeviceInfo {
    device_id: String,
    group_id: Option<String>,
    kind: MediaDeviceInfoKind,
    label: String,
    device: Box<dyn std::any::Any>,
}

impl MediaDeviceInfo {
    pub(crate) fn new(
        device_id: String,
        group_id: Option<String>,
        kind: MediaDeviceInfoKind,
        label: String,
        device: Box<dyn std::any::Any>,
    ) -> Self {
        Self {
            device_id,
            group_id,
            kind,
            label,
            device,
        }
    }

    /// Identifier for the represented device
    ///
    /// The current implementation is not stable across sessions so you should not persist this
    /// value
    pub fn device_id(&self) -> &str {
        &self.device_id
    }

    /// Two devices have the same group identifier if they belong to the same physical device
    pub fn group_id(&self) -> Option<&str> {
        self.group_id.as_deref()
    }

    /// Enumerated value that is either "videoinput", "audioinput" or "audiooutput".
    pub fn kind(&self) -> MediaDeviceInfoKind {
        self.kind
    }

    /// Friendly label describing this device
    pub fn label(&self) -> &str {
        &self.label
    }

    pub(crate) fn device(self) -> Box<dyn std::any::Any> {
        self.device
    }
}

/// Prompt for permission to use a media input (audio only)
///
/// This produces a [`MediaStream`] with tracks containing the requested types of media, which can
/// be used inside a [`MediaStreamAudioSourceNode`](crate::node::MediaStreamAudioSourceNode).
///
/// It is okay for the `MediaStream` struct to go out of scope, any corresponding stream will still be
/// kept alive and emit audio buffers. Call the `close()` method if you want to stop the media
/// input and release all system resources.
///
/// # Example
///
/// ```no_run
/// use web_audio_api::context::{BaseAudioContext, AudioContext};
/// use web_audio_api::context::{AudioContextLatencyCategory, AudioContextOptions};
/// use web_audio_api::media_devices;
/// use web_audio_api::node::AudioNode;
///
/// let context = AudioContext::default();
/// let mic = media_devices::get_user_media();
///
/// // register as media element in the audio context
/// let background = context.create_media_stream_source(&mic);
///
/// // connect the node directly to the destination node (speakers)
/// background.connect(&context.destination());
///
/// // enjoy listening
/// std::thread::sleep(std::time::Duration::from_secs(4));
/// ```
// TODO, return Promise? How to provide constraints?
pub fn get_user_media() -> MediaStream {
    crate::io::build_input(AudioContextOptions::default())
}