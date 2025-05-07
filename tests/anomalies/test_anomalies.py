from dackar.anomalies.MatrixProfile import MatrixProfile
import numpy as np

time_series = [0, 1, 3, 2, 9, 1, 14, 15, 1, 2, 2, 10, 7]
m = 4

def test_simple_anomaly():
  mp_obj = MatrixProfile(m, normalize='robust', method='normal')
  mp_obj.fit(time_series)
  data = mp_obj.get_mp()
  sol = np.asarray([0.85695683, 0.1767767, 0.77055175, 0.99215674, 1.42521928, 1.6955825 , 2.25, 1.74553001, 0.1767767 , 0.77055175])
  np.testing.assert_allclose(data[0], sol, rtol=1e-5, atol=1e-5)


def test_simple_anomaly_approx():
  mp_obj = MatrixProfile(m, normalize='robust', method='approx')
  mp_obj.fit(time_series)
  data = mp_obj.get_mp()
  sol = np.asarray([0.85695683, 0.1767767, 0.77055175, 0.99215674, 1.42521928, 1.6955825 , 2.25, 1.74553001, 0.1767767 , 0.77055175])
  np.testing.assert_allclose(data[0], sol, rtol=1e-5, atol=1e-5)


def test_simple_anomaly_streaming():
  mp_obj = MatrixProfile(m, normalize='robust', method='incremental')
  mp_obj.fit(time_series[:8])
  mp_obj.evaluate(time_series[8:])
  data = mp_obj.get_mp()
  sol = np.asarray([0.741152, 0.152888, 0.666423, 0.858082, 1.232622, 1.46645,1.945946, 1.509648, 0.152888, 0.666423])
  np.testing.assert_allclose(data[0], sol, rtol=1e-5, atol=1e-5)

def test_simple_anomaly_standard_scalar():
  mp_obj = MatrixProfile(m, normalize=None, method='normal')
  mp_obj.fit(time_series)
  data = mp_obj.get_mp()
  sol = np.asarray([0.361339, 0.280823, 1.224078, 1.576114, 2.264065, 2.693557, 3.57429, 2.772902, 0.280823, 1.224078])
  np.testing.assert_allclose(data[0], sol, rtol=1e-5, atol=1e-5)
