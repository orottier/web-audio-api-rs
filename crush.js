class Bitcrusher extends AudioWorkletProcessor {
    static get parameterDescriptors () {
        return [{
            name: 'bitDepth',
            defaultValue: 12,
            minValue: 1,
            maxValue: 16
        }, {
            name: 'frequencyReduction',
            defaultValue: 0.5,
            minValue: 0,
            maxValue: 1
        }];
    }
    constructor (options) {
        // The initial parameter value can be set by passing |options|
        // to the processor's constructor.
        super(options);
        this._phase = 0;
        this._lastSampleValue = 0;
        console.log("init");
    }
    process (inputs, outputs, parameters) {
        const input = inputs[0];
        const output = outputs[0];
        const bitDepth = parameters.bitDepth;
        const frequencyReduction = parameters.frequencyReduction;
        if (bitDepth.length > 1) {
            for (let channel = 0; channel < output.length; ++channel) {
                for (let i = 0; i < output[channel].length; ++i) {
                    let step = Math.pow(0.5, bitDepth[i]);
                    // Use modulo for indexing to handle the case where
                    // the length of the frequencyReduction array is 1.
                    this._phase += frequencyReduction[i % frequencyReduction.length];
                    if (this._phase >= 1.0) {
                        this._phase -= 1.0;
                        this._lastSampleValue =
                            step * Math.floor(input[channel][i] / step + 0.5);
                    }
                    output[channel][i] = this._lastSampleValue;
                }
            }
        } else {
            // Because we know bitDepth is constant for this call,
            // we can lift the computation of step outside the loop,
            // saving many operations.
            const step = Math.pow(0.5, bitDepth[0]);
            for (let channel = 0; channel < output.length; ++channel) {
                for (let i = 0; i < output[channel].length; ++i) {
                    this._phase += frequencyReduction[i % frequencyReduction.length];
                    if (this._phase >= 1.0) {
                        this._phase -= 1.0;
                        this._lastSampleValue =
                            step * Math.floor(input[channel][i] / step + 0.5);
                    }
                    output[channel][i] = this._lastSampleValue;
                }
            }
        }
        // No need to return a value; this node's lifetime is dependent only on its
        // input connections.
    }
}

registerProcessor('bitcrusher', Bitcrusher);
