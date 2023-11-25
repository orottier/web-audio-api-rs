# Version History

## Version 0.37.0 (2023-11-25)

- Added the AudioWorkletNode and AudioWorkletProcessor interfaces
- Added stereo-to-stereo panning for PannerNode in EqualPower mode
- Aggressively release resources of 'orphaned' audio nodes
- Tuned the volume of stereo-to-stereo panning for PannerNode in HRTF mode
- Added `context.run_diagnostics` for debugging purposes
- Fixed a panic that could occur during audio graph booting (flawed `is_active` check)
- Fix: `disconnect` would disassociate an AudioNode from its AudioParams
- Fix: properly clamp PannerNode rollOffFactor according to the distance model

## Version 0.36.1 (2023-11-08)

- Fix compilation on targets that are not x86/aarch64
- Added the ability to run benchmarks locally with `criterion`

## Version 0.36.0 (2023-10-20)

- Disable floating point denormals in audio processing via processor flags
- Do not spawn a garbage collector thread for an OfflineAudioContext
- AudioParam now derives `Clone`

## Version 0.35.0 (2023-10-18)

- Fix: panic when creating and dropping PannerNodes
- Improved performance of setting panning mode to HRTF

## Version 0.34.0 (2023-10-12)

- Breaking: many AudioNode setter methods now require `&mut self` instead of `&self`
- Fix: all audio node settings are now applied in order in the audio graph
- Fix: render thread would crash when a PannerNode is removed
- Fixed and improved device ids for audio input/output devices
- Added current playhead position for AudioBufferSourceNode

## Version 0.33.0 (2023-07-27)

- Fix: clamp to number of output channels to 32 even if the hardware supports more to prevent runtime panics
- Fix: prevent a render thread panic if the audio graph is not fully initialized yet
- Fix: rename AudioDestination max_channels_count to max_channel_count
- Fix: AudioBufferSourceNode is now Send + Sync
- Change AudioProcessor::onmessage signature to prevent deallocations
- Add garbage collector thread as sidecar of the render thread to handle some deallocations

## Version 0.32.0 (2023-07-16)

- Fix for some audio node settings being applied out of order
- Fix setting the ChannelCountMode for the OfflineAudioContext destination node
- Fix and extend AAC/M4A/ALAC decoding
- Fix broken MediaElementSourceNode for multi-channel output
- Updated Minimum Supported Rust Version (MSRV) to 1.70

## Version 0.31.0 (2023-06-25)

- Avoid allocations in Waveshaper node for real-time safety
- Improvements for the JACK audio host
- Document ALSA limitations for low latencies, and add fallback to examples
- Improve real-time safety of the render thread by using bounded channels
- Improve our usage of atomics
- Update to Rust edition 2021

## Version 0.30.0 (2023-06-07)

- Implement MediaRecorder API
- AudioContext now uses default stereo channels
- Fix issue with AudioBufferSourceNode playback rate and detune
- Rename `enumerate_devices` to `enumerate_devices_sync`
- Don't panic on unavailable input/output device selection

## Version 0.29.0 (2023-05-07)

- Implement part of the MediaStreams API (MediaStream, MediaStreamTrack)
- Implement part of the MediaDevices API (getUserMedia, enumerateDevices)
- Changed enumerateDevices to also include input devices
- Microphone input can now specify desired deviceId
- Added MediaStreamTrackAudioSourceNode
- Fixed windows build by removing termion from dev-dependencies

## Version 0.28.0 (2023-01-30)

- Improved AnalyserNode (performance and correctness)
- Fixed microphone input on Raspberry Pi
- Fixed overriding channel config for PannerNode

## Version 0.27.0 (2023-01-17)

- Head Related Transfer Function (HRTF) panning mode
- Implemented event handlers
- Prepare API for variable render quantum sizes

## Version 0.26.0 (2022-11-13)

- Added the "none" `sinkId`, render audio graph without emitting to speakers
- Fix `ConvolverNode.set_normalize` to take `&self`, not `&mut self`

## Version 0.25.0 (2022-11-06)

- Added AudioRenderCapacity functionality
- Added sinkId functionality to AudioContext (specify audio output device)
- Renamed `ChannelConfigOptions.{mode -> count_mode}`

## Version 0.24.0 (2022-09-10)

- Added ConvolverNode (mono only)
- GainNode, BiquadFilterNode, AudioParam performance improvements

## Version 0.23.0 (2022-08-23)

- AudioParam computed values array now contains only a single value when k-rate or no automations scheduled
- DelayNode supports sub-quantum delay
- IIRFilterNode and BiquadFilterNode can handle multi-channel inputs
- Various performance improvements

## Version 0.22.0 (2022-07-29)

- Added DynamicsCompressorNode
- Added `cubeb` as an alternative audio backend

## Version 0.21.0 (2022-07-23)

- Implemented MediaElement and MediaElementSourceNode
- Improved performance of AudioBufferSourceNode and DelayNode
- Make AudioContext and OfflineAudioContext `Send` and `Sync`
- Relaxed the 'balanced' and 'playback' latency - to run smoothly on RPi
- Fixes on clamping and value calculations of AudioParams
- Can now change automation rate of an AudioParam on the fly
- Implemented the concept of 'actively processing' for AudioNodes

## Version 0.20.0 (2022-07-02)

- Change sample rate type to plain f32
- Remove namespacing of buffer, audio_param and periodic_wave
- Reduce AudioRenderQuantum public API surface
- Remove ConcreteBaseAudioContext from public API docs

## Version 0.19.0 (2022-06-01)

- Added baseLatency and outputLatency attributes
- Audio processor callback now has access to AudioWorkletGlobalScope-like env
- Performance optimization for sorting large audio graphs
- Use default sample rate for output devices instead of highest to prevent insane values
- Fixed incorrect channel mixing for AudioNode input buffers

## Version 0.18.0 (2022-04-12)

- Implement BaseAudioContext state
- AudioContext can now change the number of output channels while running
- Microphone input stream is now configurable
- Microphone can properly pause, resume and close
- Consistently use `usize` for channels, inputs, lengths

## Version 0.17.0 (2022-04-03)

- Simplify AudioNode's required channel methods
- Apply all channel count/mode/interpretation constraints
- AudioContext can now update channel count while playing
- Improve AudioContext constructor
- Validate more input values for AudioParam events

## Version 0.16.0 (2022-03-20)

- AudioBufferSourceNode can now resample
- Add MediaStreamAudioDestinationNode
- Performance improvement when no PannerNodes are present
- More consistent method/argument names
- Added benchmark program
- Removed MediaElement and MediaElementSourceNode for now, will reimplement

## Version 0.15.0 (2022-02-10)

- Allow method chaining on AudioParams
- Some fallible methods will now panic instead of returning a Result
- Sub sample scheduling for OscillatorNode
- Implement AudioContext.close to free resources
- Rename BaseAudioContext trait and concrete type
- Fix spec deviations for node methods and constructor options and defaults
- Rename some functions to `_sync` to denote they do not return a Promise

## Version 0.14.0 (2022-01-13)

- Implemented context.decodeAudioData
- New media decoder using symphonia crate (with MP3 support)
- WaveShaper fixes
- Spec compliance: float for sample rate, thread safe nodes, naming of nodes

## Version 0.13.0 (2021-12-28)

- Support cyclic audio graphs with DelayNode acting as cycle breaker
- Greatly improved AudioBufferSourceNode (sub quantum scheduling, more
  controls, performance)
- Added cone gain functionality for PannerNode
- Improved performance of audio graph rendering

## Version 0.12.0 (2021-12-08)

- Support all AudioParam automation methods
- Change AudioProcessor API to match spec better

## Version 0.11.0 (2021-11-25)

- Support AudioParam.exponentialRampToValueAtTime
- Fix runaway memory and CPU usage due to unreleased nodes from graph

## Version 0.10.0 (2021-11-10)

- Added IirFilterNode
- Added AudioContextOptions
- Support more audio mixing modes

## Version 0.9.0 (2021-11-02)

- Added WaveShaperNode
- Added StereoPannerNode

## Version 0.8.0 (2021-10-21)

- OscillatorNode improvements
- Added BiquadNode
