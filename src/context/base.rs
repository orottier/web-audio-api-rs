//! The `BaseAudioContext` interface

use crate::buffer::{AudioBuffer, AudioBufferOptions};
use crate::context::{
    AudioContextRegistration, AudioNodeId, AudioParamId, ConcreteBaseAudioContext,
    DESTINATION_NODE_ID, LISTENER_PARAM_IDS,
};
use crate::media::{MediaDecoder, MediaStream};
use crate::node::{AudioNode, ChannelConfig, ChannelConfigOptions};
use crate::param::{AudioParam, AudioParamDescriptor};
use crate::periodic_wave::{PeriodicWave, PeriodicWaveOptions};
use crate::{node, AudioListener, SampleRate};

/// The interface representing an audio-processing graph built from audio modules linked together,
/// each represented by an `AudioNode`.
///
/// An audio context controls both the creation of the nodes it contains and the execution of the
/// audio processing, or decoding.
///
/// Please note that in rust, we need to differentiate between the [`BaseAudioContext`] trait and
/// the [`ConcreteBaseAudioContext`] concrete implementation.
#[allow(clippy::module_name_repetitions)]
pub trait BaseAudioContext {
    /// retrieves the `ConcreteBaseAudioContext` associated with this `AudioContext`
    fn base(&self) -> &ConcreteBaseAudioContext;

    /// Decode an [`AudioBuffer`] from a given input stream.
    ///
    /// The current implementation can decode FLAC, Opus, PCM, Vorbis, and Wav.
    ///
    /// In addition to the official spec, the input parameter can be any byte stream (not just an
    /// array). This means you can decode audio data from a file, network stream, or in memory
    /// buffer, and any other [`std::io::Read`] implementor. The data if buffered internally so you
    /// should not wrap the source in a `BufReader`.
    ///
    /// This function operates synchronously, which may be undesirable on the control thread. The
    /// example shows how to avoid this. An async version is currently not implemented.
    ///
    /// # Errors
    ///
    /// This method returns an Error in various cases (IO, mime sniffing, decoding).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::io::Cursor;
    /// use web_audio_api::SampleRate;
    /// use web_audio_api::context::{BaseAudioContext, OfflineAudioContext};
    ///
    /// let input = Cursor::new(vec![0; 32]); // or a File, TcpStream, ...
    ///
    /// let context = OfflineAudioContext::new(2, 44_100, SampleRate(44_100));
    /// let handle = std::thread::spawn(move || context.decode_audio_data_sync(input));
    ///
    /// // do other things
    ///
    /// // await result from the decoder thread
    /// let decode_buffer_result = handle.join();
    /// ```
    fn decode_audio_data_sync<R: std::io::Read + Send + Sync + 'static>(
        &self,
        input: R,
    ) -> Result<AudioBuffer, Box<dyn std::error::Error + Send + Sync>> {
        // Set up a media decoder, consume the stream in full and construct a single buffer out of it
        let mut buffer = MediaDecoder::try_new(input)?
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .reduce(|mut accum, item| {
                accum.extend(&item);
                accum
            })
            // if there are no samples decoded, return an empty buffer
            .unwrap_or_else(|| AudioBuffer::from(vec![vec![]], self.sample_rate_raw()));

        // resample to desired rate (no-op if already matching)
        buffer.resample(self.sample_rate_raw());

        Ok(buffer)
    }

    /// Create an new "in-memory" `AudioBuffer` with the given number of channels,
    /// length (i.e. number of samples per channel) and sample rate.
    ///
    /// Note: In most cases you will want the sample rate to match the current
    /// audio context sample rate.
    fn create_buffer(
        &self,
        number_of_channels: usize,
        length: usize,
        sample_rate: SampleRate,
    ) -> AudioBuffer {
        let options = AudioBufferOptions {
            number_of_channels,
            length,
            sample_rate,
        };

        AudioBuffer::new(options)
    }

    /// Creates a `AnalyserNode`
    fn create_analyser(&self) -> node::AnalyserNode {
        node::AnalyserNode::new(self.base(), node::AnalyserOptions::default())
    }

    /// Creates an `BiquadFilterNode` which implements a second order filter
    fn create_biquad_filter(&self) -> node::BiquadFilterNode {
        node::BiquadFilterNode::new(self.base(), node::BiquadFilterOptions::default())
    }

    /// Creates an `AudioBufferSourceNode`
    fn create_buffer_source(&self) -> node::AudioBufferSourceNode {
        node::AudioBufferSourceNode::new(self.base(), node::AudioBufferSourceOptions::default())
    }

    /// Creates an `ConstantSourceNode`, a source representing a constant value
    fn create_constant_source(&self) -> node::ConstantSourceNode {
        node::ConstantSourceNode::new(self.base(), node::ConstantSourceOptions::default())
    }

    /// Creates a `ChannelMergerNode`
    fn create_channel_merger(&self, number_of_inputs: u32) -> node::ChannelMergerNode {
        let opts = node::ChannelMergerOptions {
            number_of_inputs,
            ..node::ChannelMergerOptions::default()
        };
        node::ChannelMergerNode::new(self.base(), opts)
    }

    /// Creates a `ChannelSplitterNode`
    fn create_channel_splitter(&self, number_of_outputs: u32) -> node::ChannelSplitterNode {
        let opts = node::ChannelSplitterOptions {
            number_of_outputs,
            ..node::ChannelSplitterOptions::default()
        };
        node::ChannelSplitterNode::new(self.base(), opts)
    }

    /// Creates a `DelayNode`, delaying the audio signal
    fn create_delay(&self, max_delay_time: f64) -> node::DelayNode {
        let opts = node::DelayOptions {
            max_delay_time,
            ..node::DelayOptions::default()
        };
        node::DelayNode::new(self.base(), opts)
    }

    /// Creates an `GainNode`, to control audio volume
    fn create_gain(&self) -> node::GainNode {
        node::GainNode::new(self.base(), node::GainOptions::default())
    }

    /// Creates an `IirFilterNode`
    ///
    /// # Arguments
    ///
    /// * `feedforward` - An array of the feedforward (numerator) coefficients for the transfer function of the IIR filter.
    /// The maximum length of this array is 20
    /// * `feedback` - An array of the feedback (denominator) coefficients for the transfer function of the IIR filter.
    /// The maximum length of this array is 20
    fn create_iir_filter(&self, feedforward: Vec<f64>, feedback: Vec<f64>) -> node::IIRFilterNode {
        let options = node::IIRFilterOptions {
            channel_config: ChannelConfigOptions::default(),
            feedforward,
            feedback,
        };
        node::IIRFilterNode::new(self.base(), options)
    }

    /// Creates a `MediaStreamAudioSourceNode` from a [`MediaStream`]
    fn create_media_stream_source<M: MediaStream>(
        &self,
        media: M,
    ) -> node::MediaStreamAudioSourceNode {
        let opts = node::MediaStreamAudioSourceOptions {
            media_stream: media,
        };
        node::MediaStreamAudioSourceNode::new(self.base(), opts)
    }

    /// Creates a `MediaStreamAudioDestinationNode`
    fn create_media_stream_destination(&self) -> node::MediaStreamAudioDestinationNode {
        let opts = ChannelConfigOptions::default();
        node::MediaStreamAudioDestinationNode::new(self.base(), opts)
    }

    /// Creates an `OscillatorNode`, a source representing a periodic waveform.
    fn create_oscillator(&self) -> node::OscillatorNode {
        node::OscillatorNode::new(self.base(), node::OscillatorOptions::default())
    }

    /// Creates a `PannerNode`
    fn create_panner(&self) -> node::PannerNode {
        node::PannerNode::new(self.base(), node::PannerOptions::default())
    }

    /// Creates a periodic wave
    fn create_periodic_wave(&self, options: PeriodicWaveOptions) -> PeriodicWave {
        PeriodicWave::new(self.base(), options)
    }

    /// Creates an `StereoPannerNode` to pan a stereo output
    fn create_stereo_panner(&self) -> node::StereoPannerNode {
        node::StereoPannerNode::new(self.base(), node::StereoPannerOptions::default())
    }

    /// Creates a `WaveShaperNode`
    fn create_wave_shaper(&self) -> node::WaveShaperNode {
        node::WaveShaperNode::new(self.base(), node::WaveShaperOptions::default())
    }

    /// Create an `AudioParam`.
    ///
    /// Call this inside the `register` closure when setting up your `AudioNode`
    fn create_audio_param(
        &self,
        opts: AudioParamDescriptor,
        dest: &AudioNodeId,
    ) -> (crate::param::AudioParam, AudioParamId) {
        let param = self.base().register(move |registration| {
            let (node, proc) = crate::param::audio_param_pair(opts, registration);

            (node, Box::new(proc))
        });

        // Connect the param to the node, once the node is registered inside the audio graph.
        self.base().queue_audio_param_connect(&param, dest);

        let proc_id = AudioParamId(param.id().0);
        (param, proc_id)
    }

    /// Returns an `AudioDestinationNode` representing the final destination of all audio in the
    /// context. It can be thought of as the audio-rendering device.
    fn destination(&self) -> node::AudioDestinationNode {
        let registration = AudioContextRegistration {
            id: AudioNodeId(DESTINATION_NODE_ID),
            context: self.base().clone(),
        };
        let channel_count = self.base().inner.destination_channel_count.clone();
        let channel_config = ChannelConfig::for_destination(channel_count);
        node::AudioDestinationNode::from_raw_parts(registration, channel_config)
    }

    /// Returns the `AudioListener` which is used for 3D spatialization
    fn listener(&self) -> AudioListener {
        let mut ids = LISTENER_PARAM_IDS.map(|i| AudioContextRegistration {
            id: AudioNodeId(i),
            context: self.base().clone(),
        });
        let params = self.base().inner.listener_params.as_ref().unwrap();

        AudioListener {
            position_x: AudioParam::from_raw_parts(ids.next().unwrap(), params.position_x.clone()),
            position_y: AudioParam::from_raw_parts(ids.next().unwrap(), params.position_y.clone()),
            position_z: AudioParam::from_raw_parts(ids.next().unwrap(), params.position_z.clone()),
            forward_x: AudioParam::from_raw_parts(ids.next().unwrap(), params.forward_x.clone()),
            forward_y: AudioParam::from_raw_parts(ids.next().unwrap(), params.forward_y.clone()),
            forward_z: AudioParam::from_raw_parts(ids.next().unwrap(), params.forward_z.clone()),
            up_x: AudioParam::from_raw_parts(ids.next().unwrap(), params.up_x.clone()),
            up_y: AudioParam::from_raw_parts(ids.next().unwrap(), params.up_y.clone()),
            up_z: AudioParam::from_raw_parts(ids.next().unwrap(), params.up_z.clone()),
        }
    }

    /// The sample rate (in sample-frames per second) at which the `AudioContext` handles audio.
    #[must_use]
    fn sample_rate(&self) -> f32 {
        self.base().sample_rate()
    }

    /// The raw sample rate of the `AudioContext` (which has more precision than the float
    /// [`sample_rate()`](BaseAudioContext::sample_rate) value).
    #[must_use]
    fn sample_rate_raw(&self) -> SampleRate {
        self.base().sample_rate_raw()
    }

    /// This is the time in seconds of the sample frame immediately following the last sample-frame
    /// in the block of audio most recently processed by the context’s rendering graph.
    fn current_time(&self) -> f64 {
        self.base().current_time()
    }

    #[cfg(test)]
    fn mock_registration(&self) -> AudioContextRegistration {
        AudioContextRegistration {
            id: AudioNodeId(0),
            context: self.base().clone(),
        }
    }
}
