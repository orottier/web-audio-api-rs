import web_audio_api
from time import sleep

ctx = web_audio_api.AudioContext()
osc = web_audio_api.OscillatorNode(ctx)
osc.connect(ctx.destination())
osc.start()

print("freq =", osc.frequency().value);
sleep(4)

osc.frequency().value = 300
print("freq =", osc.frequency().value);

sleep(4)
