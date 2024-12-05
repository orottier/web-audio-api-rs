# Python bindings for web-audio-api-rs

## Local development

```bash
# cd to this directory

# if not already, create a virtual env
python3 -m venv .env

# enter the virtual env
source .env/bin/activate

# (re)build the package
maturin develop
```

```python
import web_audio_api
ctx = web_audio_api.AudioContext()
osc = web_audio_api.OscillatorNode(ctx)
osc.connect(ctx.destination())
osc.start()
osc.frequency().set_value(300)
```
