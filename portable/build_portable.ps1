<#
Builds the portable "ML Forecasting Pipeline" folder on a drive.

  powershell -ExecutionPolicy Bypass -File portable\build_portable.ps1 -Target E:\MLForecastPipeline

Layout written to -Target:
  ML Forecasting Pipeline.exe   launcher (C#, compiled with the .NET Framework csc in Windows)
  runtime\python\               relocatable CPython 3.12 (python-build-standalone) + packages + VC runtime DLLs
  runtime\wheels\               offline wheelhouse + requirements.lock (repairs the runtime on any PC)
  app\gui\                      pipeline_gui.py, pipeline_tasks.py, assets\
  app\repo\                     pipeline scripts and data
  models\  logs\  work\tmp\     outputs (models are exported here, on the drive)
  config.json                   optional keys passed to the run (e.g. WU_API_KEY)
#>
param(
    [string]$Target = "E:\MLForecastPipeline",
    [string]$PythonUrl = "https://github.com/astral-sh/python-build-standalone/releases/download/20261013/cpython-3.12.15%2B20261013-x86_64-pc-windows-msvc-install_only.tar.gz",
    [switch]$SkipWheels,
    [switch]$CodeOnly
)
$ErrorActionPreference = "Stop"
$Here = $PSScriptRoot
$Repo = Split-Path -Parent $Here
$Cache = Join-Path $Here "cache"
$Py = Join-Path $Target "runtime\python\python.exe"

function Step($msg) { Write-Host ""; Write-Host "== $msg" -ForegroundColor Cyan }

foreach ($d in @($Target, "$Target\runtime", "$Target\app\gui", "$Target\app\repo", "$Target\models",
                 "$Target\logs", "$Target\work\tmp", $Cache)) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}

if (-not $CodeOnly) {
    Step "Python runtime"
    if (-not (Test-Path $Py)) {
        $tgz = Join-Path $Cache ([System.Uri]::UnescapeDataString(($PythonUrl -split "/")[-1]))
        if (-not (Test-Path $tgz)) {
            Write-Host "Downloading $PythonUrl"
            Invoke-WebRequest -Uri $PythonUrl -OutFile $tgz -UseBasicParsing
        }
        tar -xzf $tgz -C "$Target\runtime"
        if ($LASTEXITCODE -ne 0) { throw "tar failed" }
    }
    & $Py --version

    Step "Packages"
    & $Py -m pip install --disable-pip-version-check --no-warn-script-location -r "$Repo\requirements.txt" pillow
    if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

    if (-not $SkipWheels) {
        Step "Offline wheelhouse"
        New-Item -ItemType Directory -Force -Path "$Target\runtime\wheels" | Out-Null
        & $Py -m pip download --disable-pip-version-check -r "$Repo\requirements.txt" pillow -d "$Target\runtime\wheels"
        if ($LASTEXITCODE -ne 0) { throw "pip download failed" }
        & $Py -m pip freeze --disable-pip-version-check | Out-File -Encoding ascii "$Target\runtime\wheels\requirements.lock"
    }

    Step "VC++ runtime DLLs (xgboost / lightgbm / catboost)"
    foreach ($dll in @("vcomp140.dll", "msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll",
                       "vcruntime140.dll", "vcruntime140_1.dll", "concrt140.dll")) {
        $src = Join-Path $env:WINDIR "System32\$dll"
        $dst = Join-Path $Target "runtime\python\$dll"
        if ((Test-Path $src) -and -not (Test-Path $dst)) { Copy-Item $src $dst }
    }
}

Step "GUI"
$venvPy = Join-Path $Repo ".venv\Scripts\python.exe"
if (Test-Path $venvPy) {
    & $venvPy "$Here\make_pressure_map.py" "$Here\gui\assets"
}
robocopy "$Here\gui" "$Target\app\gui" /E /XD __pycache__ /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy gui failed ($LASTEXITCODE)" }

Step "Launcher"
$csc = Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"
& $csc /nologo /target:winexe /platform:x64 /optimize+ `
    "/win32icon:$Target\app\gui\assets\app.ico" /r:System.Windows.Forms.dll `
    "/out:$Target\ML Forecasting Pipeline.exe" "$Here\launcher.cs"
if ($LASTEXITCODE -ne 0) { throw "csc failed" }

Step "Pipeline code and data"
robocopy $Repo "$Target\app\repo" /E /XD .git .venv artifacts portable build __pycache__ node_modules .claude cloudflare `
    /XF *.pyc /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy repo failed ($LASTEXITCODE)" }
$commit = (git -C $Repo rev-parse --short HEAD) 2>$null
$dirty = [bool]((git -C $Repo status --porcelain) 2>$null)
@{ commit = "$commit"; dirty = $dirty; built_at = (Get-Date).ToString("s"); builder = $env:COMPUTERNAME } |
    ConvertTo-Json | Out-File -Encoding utf8 "$Target\app\repo\BUILD_INFO.json"

if (-not (Test-Path "$Target\config.json")) {
    '{ "WU_API_KEY": "" }' | Out-File -Encoding ascii "$Target\config.json"
}
Copy-Item "$Here\README.txt" "$Target\README.txt" -Force

Step "Check"
& $Py -c "import numpy, pandas, xgboost, lightgbm, catboost, optuna, sklearn, PIL, tkinter; print('imports ok')"
if ($LASTEXITCODE -ne 0) { throw "runtime import check failed" }
$size = (Get-ChildItem $Target -Recurse -File | Measure-Object Length -Sum).Sum / 1GB
Write-Host ("Built {0} ({1:N2} GB)" -f $Target, $size) -ForegroundColor Green
