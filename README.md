# Rust Web Audio API

[![crates.io](https://img.shields.io/crates/v/web-audio-api.svg)](https://crates.io/crates/web-audio-api)

A Rust implementation of the Web Audio API, for use in non-browser contexts

## About the Web Audio API

[MDN docs](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API)

The Web Audio API provides a powerful and versatile system for controlling
audio on the Web, allowing developers to choose audio sources, add effects to
audio, create audio visualizations, apply spatial effects (such as panning) and
much more.

## Usage

[Docs](https://docs.rs/web-audio-api)

## Planning

- v0.\*: getting the basics out, stabilize API
- v1.\*: feature completeness
- v2.\*: full spec compliance

## Spec compliance

Current deviations

- function names use snake_case
- getters/setters instead of exposed attributes
- deprecated functions are not implemented
- some control-render communication is done with atomics instead of message passing
- function that should return Promises are now blocking
- no AudioWorklet functionality, users should implement the relevant traits instead
- ...

## Contributing

Crossbeam welcomes contribution from everyone in the form of suggestions, bug reports,
pull requests, and feedback. 💛

If you need ideas for contribution, there are several ways to get started:

- Found a bug or have a feature request?
  [Submit an issue](https://github.com/orottier/web-audio-api-rs/issues/new)!
- Issues and PRs labeled with
  [feedback wanted](https://github.com/orottier/web-audio-api-rs/issues?utf8=%E2%9C%93&q=is%3Aopen+sort%3Aupdated-desc+label%3A%22feedback+wanted%22+)
  need feedback from users and contributors.
- Issues labeled with
  [good first issue](https://github.com/orottier/web-audio-api-rs/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc+label%3A%22good+first+issue%22)
  are relatively easy starter issues.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in web-audio-api-rs by you, shall be licensed as MIT, without any additional
terms or conditions.

## License

This project is licensed under the [MIT license].

[mit license]: https://github.com/orottier/web-audio-api-rs/blob/main/LICENSE
