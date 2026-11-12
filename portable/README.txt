ML Forecasting Pipeline by Matija Ivanovic(TM)
==============================================

Run "ML Forecasting Pipeline.exe" from this folder on any 64-bit Windows 10/11 PC.
Nothing is installed on that PC: Python and every package are in runtime\.

The Setup Wizard: Welcome > Run Options (device, Optuna trials) > Ready > Run.
It then runs, in order:
  1. Checking prerequisites   bundled Python and packages (repaired from
                              runtime\wheels if needed), free space, network, GPU
  2. Downloading station data IBUDVA5 up to now
  3. Downloading model data   forecast archives, previous runs, upper-air history
  4. Rain feature backtest    picks the rain classifier's feature set
  5. Training models          all targets + rain classifier, then a live forecast
  6. Running tests
  7. Saving the model         models\<date_time>\ with MANIFEST.json

Everything is written to this drive: the trained model (models\), logs (logs\),
temporary files (work\tmp\). The drive letter can be anything.

Needs an internet connection for steps 2-3 and the live forecast. With an
NVIDIA GPU and driver, "Automatic" trains on the GPU, otherwise on the CPU
(a CPU run takes several hours; keep the PC plugged in, it will not sleep).

config.json: optional settings passed to the run, e.g. your own
Weather Underground key: { "WU_API_KEY": "..." }.
