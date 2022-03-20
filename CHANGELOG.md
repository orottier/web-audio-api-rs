# Version 0.16.0 (2021-03-20)

- AudioBufferSourceNode can now resample
- Add MediaStreamAudioDestinationNode
- Performance improvement when no PannerNodes are present
- More consistent method/argument names
- Added benchmark program
- Removed MediaElement and MediaElementSourceNode for now, will reimplement

# Version 0.15.0 (2021-02-10)

- Allow method chaining on AudioParams
- Some fallible methods will now panic instead of returning a Result
- Sub sample scheduling for OscillatorNode
- Implement AudioContext.close to free resources
- Rename BaseAudioContext trait and concrete type
- Fix spec deviations for node methods and constructor options and defaults
- Rename some functions to `_sync` to denote they do not return a Promise

# Version 0.14.0 (2021-01-13)

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
