import numpy as np
from scipy import signal as sig

def binaryProfile(freqFact=None,
                  sigmaAnomaly=None,
                  ampAnomaly=None,
                  nullLoc=None, nullDur=None,
                  offsetVar=None,
                  impLoc=None,    impVal=None):
    """
        This method generates a periodic square wave function with noise included.
        Orignal parameteres of this function are:
        * frequency: np.pi * 2
        * noise: sigma=.03

        Args:

        freqFact: float, frequency multiplier for an anomaly in the wave frequency

        sigmaAnomaly: float, sigma for an anomaly in the wave noise

        ampAnomaly: float, multiplier for an anomaly in the wave amplitude (prior noise addition)

        nullLoc: integer, location in the [0,500] range for an anomaly where signal is set to 0
        nullDur: integer, duration value in the [0,500] range for an anomaly where signal is set to 0

        offsetVar: float, vertical offset for an anomaly in the wave signal

        impVal: float, value for a point-value anomaly in the wave noise
        impLoc: integer, location in the [0,500] range for a point-value anomaly where signal value is set to impVal

        Returns:

        data: np.array, data for a single instance of a periodic square wave function
    """

    t = np.linspace(0, 3, 500, endpoint=False)

    if freqFact is None:
        data = sig.square(np.pi * 2 * t)
    else:
        data = sig.square(np.pi * 2 * freqFact * t)

    if ampAnomaly is not None:
        data = data * ampAnomaly

    if sigmaAnomaly is not None:
        noise = np.random.normal(0,sigmaAnomaly,500)
    else:
        noise = np.random.normal(0,.03,500)

    data = data + noise

    if nullLoc is not None:
        data[nullLoc:(nullLoc+nullDur)] = 0

    if offsetVar is not None:
        data = data + offsetVar

    if impLoc is not None:
        data[impLoc] = data[impLoc] + impVal

    return data


def impulse(fc, bw, sigmaNoise=.05,
            offset=None,
            impLoc=None,    impVal=None,
            offsetLoc=None, offsetDur=None, offsetVar=None,
            noiseLoc=None,  noiseDur=None,
            freezeLoc=None, freezeDur=None,
            nullLoc=None,   nullDur=None):
    """
        This method generates a Gaussian modulated sinusoid pulse function with noise included.
        Orignal parameteres of this function are:
        * frequency: np.pi * 2
        * noise: sigma=.05

        Args:

        offset: float, vertical offset for an anomaly in the pulse signal

        impVal: float, value for a point-value anomaly in the pulse signal
        impLoc: integer, location in the [0,500] range for a point-value anomaly where signal value is set to impVal

        offsetVar: float, vertical offset for an anomaly in the pulse  signal
        offsetLoc: integer, location in the [0,200] range for a vertical offset anomaly
        offsetDur: integer, duration in the [0,200] range for a vertical offset anomaly

        noiseLoc: integer, location in the [0,200] range for an anomaly where noise is multiplied by 2.0
        noiseDur: integer, duration in the [0,200] range for a noise anomaly

        freezeLoc: integer, location in the [0,200] range for an anomaly where signal is kept constant to the previous signal value (freeze)
        freezeDur: integer, duration in the [0,200] range for a freeze anomaly

        nullLoc: integer, location in the [0,200] range for an anomaly where signal is set to 0
        nullDur: integer, duration value in the [0,200] range for an anomaly where signal is set to 0

        Returns:

        data: np.array, data for a single instance of a Gaussian modulated sinusoid pulse function
    """

    t = np.linspace(1.5, -1.5, 200, endpoint=False)

    impulse = sig.gausspulse(t, fc, bw)

    if freezeLoc is not None:
        impulse[freezeLoc:(freezeLoc+freezeDur)] = impulse[freezeLoc]

    if offset is not None:
        impulse = impulse + offset

    noise = np.random.normal(0, sigmaNoise, 200)

    if noiseLoc is not None:
        noise[noiseLoc:(noiseLoc+noiseDur)] = noise [noiseLoc:(noiseLoc+noiseDur)] * 2.0

    signal = impulse + noise

    if impLoc is not None:
        signal[impLoc] = signal[impLoc] + impVal

    if offsetLoc is not None:
        signal[offsetLoc:(offsetLoc+offsetDur)] = signal[offsetLoc:(offsetLoc+offsetDur)] + offsetVar

    if nullLoc is not None:
        signal[nullLoc:(nullLoc+nullDur)] = 0.0

    return signal


def yearProfile(impLoc=None,    impFac=None,
                freezeLoc=None, freezeDur=None,
                noiseLoc=None,  noiseDur=None,
                offsetLoc=None, offsetDur=None,
                driftLoc=None,  driftDur= None, driftFact=None):
    """
        This method generates a sinusoidal function with noise included that emulates yearly seasonality.
        Orignal parameteres of this function are:
        * frequency: np.pi * 2
        * noise: sigma = .05 for the first half of the signal
                 sigma = .25 for the second half of the signal

        Args:

        freezeLoc: integer, location in the [0,365] range for an anomaly where signal is kept constant to the previous signal value (freeze)
        freezeDur: integer, duration in the [0,365] range for a freeze anomaly

        noiseLoc: integer, location in the [0,365] range for an anomaly where noise is multiplied by 2.4
        noiseDur: integer, duration in the [0,365] range for a noise anomaly

        offsetLoc: integer, location in the [0,365] range for a vertical offset anomaly of .2
        offsetDur: integer, duration in the [0,365] range for a vertical offset anomaly

        driftLoc: integer, location in the [0,365] range for a drift anomaly
        driftDur: integer, duration in the [0,365] range for a drift anomaly
        driftFact: float, coefficient for the drift anomaly

        Returns:

        data: np.array, data for a single instance of a periodic square wave function
    """
    length = 365
    t = np.arange(length)
    signal = -np.sin(2 * np.pi * t / length)

    if driftLoc is not None:
        signal[driftLoc:(driftLoc+driftDur)] = signal[driftLoc:(driftLoc+driftDur)] + np.arange(0, driftDur, 1)/driftFact

    if offsetLoc is not None:
        signal[offsetLoc:(offsetLoc+offsetDur)] = signal[offsetLoc:(offsetLoc+offsetDur)] + .2

    noise = np.random.normal(0,.05,length) + signal * np.heaviside(signal,0) * np.random.normal(0,.2,length)

    if noiseLoc is not None:
        noise[noiseLoc:(noiseLoc+noiseDur)] = noise[noiseLoc:(noiseLoc+noiseDur)]*2.4

    signal = signal + noise

    if impLoc is not None:
        signal[impLoc] = signal[impLoc]*impFac

    if freezeLoc is not None:
        signal[freezeLoc:(freezeLoc+freezeDur)] = signal[freezeLoc-1]

    return signal
