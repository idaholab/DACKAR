import numpy as np
from scipy import signal as sig

def binaryProfile(freqFact=None,
                  sigmaAnomaly=None,
                  ampAnomaly=None,
                  nullLoc=None, nullDur=None):

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

    return data  

def impulse(fc, bw, sigmaNoise=.05, 
            offset=None,     
            impLoc=None,    impVal=None,
            offsetLoc=None, offsetDur=None, offsetVar=None,
            noiseLoc=None,  noiseDur=None,
            freezeLoc=None, freezeDur=None,
            nullLoc=None,   nullDur=None):

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

def HS(x):
    return np.heaviside(x, 0)

def yearProfile(impLoc=None,    impFac=None, 
                freezeLoc=None, freezeDur=None, 
                noiseLoc=None,  noiseDur=None,
                offsetLoc=None, offsetDur=None,
                driftLoc=None,  driftDur= None, driftFact=None):

    length = 365
    t = np.arange(length)
    signal = -np.sin(2 * np.pi * t / length)

    if driftLoc is not None:
        signal[driftLoc:(driftLoc+driftDur)] = signal[driftLoc:(driftLoc+driftDur)] + np.arange(0, driftDur, 1)/driftFact

    if offsetLoc is not None:
        signal[offsetLoc:(offsetLoc+offsetDur)] = signal[offsetLoc:(offsetLoc+offsetDur)] + .2

    noise = np.random.normal(0,.05,length) + signal*HS(signal)*np.random.normal(0,.2,length)

    if noiseLoc is not None:
        noise[noiseLoc:(noiseLoc+noiseDur)] = noise[noiseLoc:(noiseLoc+noiseDur)]*2.4

    signal = signal + noise

    if impLoc is not None:
        signal[impLoc] = signal[impLoc]*impFac
    
    if freezeLoc is not None:
        signal[freezeLoc:(freezeLoc+freezeDur)] = signal[freezeLoc-1]
    
    return signal