param([switch]$Setup)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$distro = 'Ubuntu-22.04'
function Run-WSL([string[]]$Arguments) {
    & wsl.exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "WSL command failed (exit $LASTEXITCODE)." }
}
# Windows PowerShell 5.1 can strip embedded quotes in native arguments.
# Transport shell scripts as base64 so neither Windows nor Bash reinterprets paths.
function Run-LinuxScript([string]$Script) {
    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($Script))
    Run-WSL -Arguments @('-d', $distro, '--exec', '/bin/bash', '-c', "echo $encoded|base64 -d|bash")
}
try {
    if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
        throw 'WSL is unavailable. Install Windows updates, then run wsl --install -d Ubuntu-22.04 in an administrator terminal and restart Windows.'
    }
    $distros = ((& wsl.exe --list --quiet) -replace "`0", '')
    if ($distros -notcontains $distro) {
        Write-Host 'Installing Ubuntu for Pollen Studio. Windows may request administrator permission.'
        Run-WSL -Arguments @('--install', '--distribution', $distro)
        Write-Host 'If Windows requested a restart, restart now. Finish Ubuntu user setup, then run Setup-Windows.cmd again.'
        exit 0
    }
    $versions = ((& wsl.exe --list --verbose) -replace "`0", '')
    $entry = $versions | Where-Object { $_ -match 'Ubuntu-22.04' }
    if ($entry -notmatch '\s2\s*$') {
        Write-Host 'Converting Ubuntu to WSL2 for GPU support…'
        Run-WSL -Arguments @('--set-version', $distro, '2')
    }
    # Convert the checkout path without interpolating it into shell source.
    $windowsSource = $root.Replace('\', '/')
    $linuxSource = & wsl.exe -d $distro --exec /usr/bin/wslpath -a -u $windowsSource
    if ($LASTEXITCODE -ne 0) { throw 'Cannot access this folder from WSL.' }
    $linuxSource = ($linuxSource -replace "`0", '').Trim()
    if ($Setup) {
        Run-LinuxScript -Script 'set -e; sudo apt-get update; sudo apt-get install -y curl ca-certificates'
        # Transfer source only; datasets and environments stay in Linux storage.
        $sourceEncoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($linuxSource))
        $copyScript = @'
set -euo pipefail
source_path=$(echo __SOURCE_BASE64__ | base64 -d)
mkdir -p "$HOME/PollenStudio"
cd "$source_path"
tar --exclude=./.venv --exclude=./.tools --exclude=./.python --exclude=./studio_data --exclude=./models --exclude=./captures --exclude=./.git --exclude=./.env --exclude='*.zip' -cf - . | tar -xf - -C "$HOME/PollenStudio"
cd "$HOME/PollenStudio"
bash Setup-Linux.sh
'@
        Run-LinuxScript -Script $copyScript.Replace('__SOURCE_BASE64__', $sourceEncoded)
        Write-Host 'Setup complete. Double-click Start-Windows.cmd.'
    } else {
        Write-Host 'Open http://localhost:8765 in your Windows browser. Keep this terminal open.'
        Run-LinuxScript -Script 'set -e; cd "$HOME/PollenStudio"; bash Start-Linux.sh'
    }
} catch {
    Write-Host $_ -ForegroundColor Red
    exit 1
}
