# Load MSVC into the current PowerShell session (so pip can find cl.exe).
# Run from repo root before: pip install -r requirements.txt
#   . .\scripts\init_msvc_env.ps1

$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    Write-Error "vswhere.exe not found. Install 'Build Tools for Visual Studio' with the C++ workload."
    return
}

$installPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $installPath) {
    Write-Error "No MSVC toolset found. In Visual Studio Installer, enable 'Desktop development with C++' (or C++ build tools)."
    return
}

$vsDevCmd = Join-Path $installPath "Common7\Tools\VsDevCmd.bat"
if (-not (Test-Path $vsDevCmd)) {
    Write-Error "VsDevCmd.bat not found at: $vsDevCmd"
    return
}

cmd /c "`"$vsDevCmd`" -arch=amd64 -host_arch=amd64 >nul && set" | ForEach-Object {
    if ($_ -match "^([^=]+)=(.*)$") {
        Set-Item -Path "Env:\$($matches[1])" -Value $matches[2]
    }
}

# PyTorch cpp_extension refuses to build if VC is on PATH but this is unset (raises UserWarning as failure).
$env:DISTUTILS_USE_SDK = "1"

Write-Host "MSVC environment loaded from: $installPath" -ForegroundColor Green
Write-Host "DISTUTILS_USE_SDK=1 (required for torch.utils.cpp_extension on Windows)" -ForegroundColor DarkGray
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    Write-Warning "cl.exe still not on PATH; open 'x64 Native Tools Command Prompt for VS 2022' and run pip there."
}
