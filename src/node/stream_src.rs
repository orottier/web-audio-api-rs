use crate::{
    buffer::{AudioBuffer, AudioBufferOptions, ChannelConfig, ChannelConfigOptions, Resampler},
    context::{AsBaseAudioContext, AudioContextRegistration},
    control::{Controller, Scheduler},
    media::{MediaElement, MediaStream},
    process::{AudioParamValues, AudioProcessor},
    BufferDepletedError, BUFFER_SIZE,
};

use super::{AudioControllableSourceNode, AudioNode, AudioScheduledSourceNode};

/// Options for constructing a MediaStreamAudioSourceNode
pub struct MediaStreamAudioSourceNodeOptions<M> {
    pub media: M,
    pub channel_config: ChannelConfigOptions,
}

/// An audio source from a [`MediaStream`] (e.g. microphone input)
///
/// IMPORTANT: the media stream is polled on the render thread so you must ensure the media stream
/// iterator never blocks. Consider wrapping the `MediaStream` in a [`MediaElement`], which buffers the
/// stream on another thread so the render thread never blocks.
pub struct MediaStreamAudioSourceNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
}

impl AudioNode for MediaStreamAudioSourceNode {
    fn registration(&self) -> &AudioContextRegistration {
        &self.registration
    }

    fn channel_config_raw(&self) -> &ChannelConfig {
        &self.channel_config
    }

    fn number_of_inputs(&self) -> u32 {
        0
    }
    fn number_of_outputs(&self) -> u32 {
        1
    }
}

impl MediaStreamAudioSourceNode {
    pub fn new<C: AsBaseAudioContext, M: MediaStream>(
        context: &C,
        options: MediaStreamAudioSourceNodeOptions<M>,
    ) -> Self {
        context.base().register(move |registration| {
            let node = MediaStreamAudioSourceNode {
                registration,
                channel_config: options.channel_config.into(),
            };

            let resampler =
                Resampler::new(context.base().sample_rate(), BUFFER_SIZE, options.media);

            // setup void scheduler - always on
            let scheduler = Scheduler::new();
            scheduler.start_at(0.);

            let render = MediaStreamRenderer::new(resampler, scheduler);

            (node, Box::new(render))
        })
    }
}

/// Options for constructing a MediaElementAudioSourceNode
pub struct MediaElementAudioSourceNodeOptions {
    pub media: MediaElement,
    pub channel_config: ChannelConfigOptions,
}

/// An audio source from a [`MediaElement`] (e.g. .ogg, .wav, .mp3 files)
///
/// The media element will take care of buffering of the stream so the render thread never blocks.
/// This also allows for playback controls (pause, looping, playback rate, etc.)
///
/// Note: do not forget to `start()` the node.
pub struct MediaElementAudioSourceNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
    controller: Controller,
}

impl AudioScheduledSourceNode for MediaElementAudioSourceNode {
    fn scheduler(&self) -> &Scheduler {
        self.controller.scheduler()
    }
}
impl AudioControllableSourceNode for MediaElementAudioSourceNode {
    fn controller(&self) -> &Controller {
        &self.controller
    }
}

impl AudioNode for MediaElementAudioSourceNode {
    fn registration(&self) -> &AudioContextRegistration {
        &self.registration
    }
    fn channel_config_raw(&self) -> &ChannelConfig {
        &self.channel_config
    }

    fn number_of_inputs(&self) -> u32 {
        0
    }
    fn number_of_outputs(&self) -> u32 {
        1
    }
}

impl MediaElementAudioSourceNode {
    pub fn new<C: AsBaseAudioContext>(
        context: &C,
        options: MediaElementAudioSourceNodeOptions,
    ) -> Self {
        context.base().register(move |registration| {
            let controller = options.media.controller().clone();
            let scheduler = controller.scheduler().clone();

            let node = MediaElementAudioSourceNode {
                registration,
                channel_config: options.channel_config.into(),
                controller,
            };

            let resampler =
                Resampler::new(context.base().sample_rate(), BUFFER_SIZE, options.media);
            let render = MediaStreamRenderer::new(resampler, scheduler);

            (node, Box::new(render))
        })
    }
}

struct MediaStreamRenderer<R> {
    stream: R,
    scheduler: Scheduler,
    finished: bool,
}

impl<R> MediaStreamRenderer<R> {
    fn new(stream: R, scheduler: Scheduler) -> Self {
        Self {
            stream,
            scheduler,
            finished: false,
        }
    }
}

impl<R: MediaStream> AudioProcessor for MediaStreamRenderer<R> {
    fn process(
        &mut self,
        _inputs: &[crate::alloc::AudioBuffer],
        outputs: &mut [crate::alloc::AudioBuffer],
        _params: AudioParamValues,
        timestamp: f64,
        _sample_rate: f32,
    ) {
        // single output node
        let output = &mut outputs[0];

        // todo, sub-quantum start/stop
        if !self.scheduler.is_active(timestamp) {
            output.make_silent();
            return;
        }

        match self.stream.next() {
            Some(Ok(buffer)) => {
                let channels = buffer.number_of_channels();
                output.set_number_of_channels(channels);
                output
                    .channels_mut()
                    .iter_mut()
                    .zip(buffer.channels())
                    .for_each(|(o, i)| o.copy_from_slice(&i[..]));
            }
            Some(Err(e)) if e.is::<BufferDepletedError>() => {
                log::debug!("media element buffer depleted");
                output.make_silent()
            }
            Some(Err(e)) => {
                log::warn!("Error playing audio stream: {}", e);
                self.finished = true; // halt playback
                output.make_silent()
            }
            None => {
                if !self.finished {
                    log::debug!("Stream finished");
                    self.finished = true;
                }
                output.make_silent()
            }
        }
    }

    fn tail_time(&self) -> bool {
        !self.finished
    }
}

/// Options for constructing a AudioBufferSourceNode
pub struct AudioBufferSourceNodeOptions {
    pub buffer: Option<AudioBuffer>,
    pub channel_config: ChannelConfigOptions,
}

impl Default for AudioBufferSourceNodeOptions {
    fn default() -> Self {
        Self {
            buffer: None,
            channel_config: ChannelConfigOptions::default(),
        }
    }
}

/// An audio source from an in-memory audio asset in an AudioBuffer
///
/// Note: do not forget to `start()` the node.
pub struct AudioBufferSourceNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
    controller: Controller,
}

impl AudioScheduledSourceNode for AudioBufferSourceNode {
    fn scheduler(&self) -> &Scheduler {
        self.controller.scheduler()
    }
}
impl AudioControllableSourceNode for AudioBufferSourceNode {
    fn controller(&self) -> &Controller {
        &self.controller
    }
}

impl AudioNode for AudioBufferSourceNode {
    fn registration(&self) -> &AudioContextRegistration {
        &self.registration
    }

    fn channel_config_raw(&self) -> &ChannelConfig {
        &self.channel_config
    }

    fn number_of_inputs(&self) -> u32 {
        0
    }
    fn number_of_outputs(&self) -> u32 {
        1
    }
}

impl AudioBufferSourceNode {
    pub fn new<C: AsBaseAudioContext>(context: &C, options: AudioBufferSourceNodeOptions) -> Self {
        context.base().register(move |registration| {
            // unwrap_or_default buffer
            let buffer = options.buffer.unwrap_or_else(|| {
                AudioBuffer::new(AudioBufferOptions {
                    number_of_channels: Some(1),
                    length: BUFFER_SIZE as usize,
                    sample_rate: 44_100.,
                })
            });

            // wrap input in resampler
            let resampler = Resampler::new(
                context.base().sample_rate(),
                BUFFER_SIZE,
                std::iter::once(Ok(buffer)),
            );

            // wrap resampler in media-element (for loop/play/pause)
            let media = MediaElement::new(resampler);
            let controller = media.controller().clone();
            let scheduler = controller.scheduler().clone();

            // setup user facing audio node
            let node = AudioBufferSourceNode {
                registration,
                channel_config: options.channel_config.into(),
                controller,
            };

            let render = MediaStreamRenderer::new(media, scheduler);

            (node, Box::new(render))
        })
    }
}
