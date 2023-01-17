# Version 0.27.0 (2023-01-17)

- Head Related Transfer Function (HRTF) panning mode
- Implemented event handlers
- Prepare API for variable render quantum sizes

# Version 0.26.0 (2022-11-13)

- Added the "none" `sinkId`, render audio graph without emitting to speakers
- Fix `ConvolverNode.set_normalize` to take `&self`, not `&mut self`

# Version 0.25.0 (2022-11-06)

- Added AudioRenderCapacity functionality
- Added sinkId functionality to AudioContext (specify audio output device)
- Renamed `ChannelConfigOptions.{mode -> count_mode}`

# Version 0.24.0 (2022-09-10)

- Added ConvolverNode (mono only)
- GainNode, BiquadFilterNode, AudioParam performance improvements

# Version 0.23.0 (2022-08-23)

- AudioParam computed values array now contains only a single value when k-rate or no automations scheduled
- DelayNode supports sub-quantum delay
- IIRFilterNode and BiquadFilterNode can handle multi-channel inputs
- Various performance improvements

# Version 0.22.0 (2022-07-29)

- Added DynamicsCompressorNode
- Added `cubeb` as an alternative audio backend

# Version 0.21.0 (2022-07-23)

- Implemented MediaElement and MediaElementSourceNode
- Improved performance of AudioBufferSourceNode and DelayNode
- Make AudioContext and OfflineAudioContext `Send` and `Sync`
- Relaxed the 'balanced' and 'playback' latency - to run smoothly on RPi
- Fixes on clamping and value calculations of AudioParams
- Can now change automation rate of an AudioParam on the fly
- Implemented the concept of 'actively processing' for AudioNodes

# Version 0.20.0 (2022-07-02)

- Change sample rate type to plain f32
- Remove namespacing of buffer, audio\_param and periodic\_wave
- Reduce AudioRenderQuantum public API surface
- Remove ConcreteBaseAudioContext from public API docs

# Version 0.19.0 (2022-06-01)

- Added baseLatency and outputLatency attributes
- Audio processor callback now has access to AudioWorkletGlobalScope-like env
- Performance optimization for sorting large audio graphs
- Use default sample rate for output devices instead of highest to prevent insane values
- Fixed incorrect channel mixing for AudioNode input buffers

# Version 0.18.0 (2022-04-12)

- Implement BaseAudioContext state
- AudioContext can now change the number of output channels while running
- Microphone input stream is now configurable
- Microphone can properly pause, resume and close
- Consistenly use `usize` for channels, inputs, lengths

# Version 0.17.0 (2022-04-03)

- Simplify AudioNode's required channel methods
- Apply all channel count/mode/interpretation constraints
- AudioContext can now update channel count while playing
- Improve AudioContext constructor
- Validate more input values for AudioParam events

# Version 0.16.0 (2022-03-20)

- AudioBufferSourceNode can now resample
- Add MediaStreamAudioDestinationNode
- Performance improvement when no PannerNodes are present
- More consistent method/argument names
- Added benchmark program
- Removed MediaElement and MediaElementSourceNode for now, will reimplement

# Version 0.15.0 (2022-02-10)

- Allow method chaining on AudioParams
- Some fallible methods will now panic instead of returning a Result
- Sub sample scheduling for OscillatorNode
- Implement AudioContext.close to free resources
- Rename BaseAudioContext trait and concrete type
- Fix spec deviations for node methods and constructor options and defaults
- Rename some functions to `_sync` to denote they do not return a Promise

# Version 0.14.0 (2022-01-13)

- Implemented context.decodeAudioData
- New media decoder using symphonia crate (with MP3 support)
- WaveShaper fixes
- Spec compliance: float for sample rate, thread safe nodes, naming of nodes

# Version 0.13.0 (2021-12-28)

- Support cyclic audio graphs with DelayNode acting as cycle breaker
- Greatly improved AudioBufferSourceNode (sub quantum scheduling, more
  controls, performance)
- Added cone gain functionality for PannerNode
- Improved performance of audio graph rendering

# Version 0.12.0 (2021-12-08)

- Support all AudioParam automation methods
- Change AudioProcessor API to match spec better

# Version 0.11.0 (2021-11-25)

- Support AudioParam.exponentialRampToValueAtTime
- Fix runaway memory and CPU usage due to unreleased nodes from graph

# Version 0.10.0 (2021-11-10)

- Added IirFilterNode
- Added AudioContextOptions
- Support more audio mixing modes

# Version 0.9.0 (2021-11-02)

- Added WaveShaperNode
- Added StereoPannerNode

# Version 0.8.0 (2021-10-21)

- OscillatorNode improvements
- Added BiquadNode
