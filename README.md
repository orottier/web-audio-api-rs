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
- v1.\*: feature completeness, performance
- v2.\*: full spec compliance

## Spec compliance

Current deviations

- function names use snake\_case
- getters/setters instead of exposed attributes
- deprecated functions are not implemented
- some control-render communication is done with atomics instead of message passing
- function that should return Promises are now blocking
- no AudioWorklet functionality, users should implement the relevant traits instead
- control messages and audio param changes may arrive out of order
- ...
