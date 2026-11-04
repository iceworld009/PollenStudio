param([switch]$Setup)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$distro = 'Ubuntu-22.04'
function Run-WSL([string[]]$Arguments) {
    & wsl.exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "WSL command failed (exit $LASTEXITCODE)." }
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
    $linuxSource = & wsl.exe -d $distro -- wslpath -a -u $root
    if ($LASTEXITCODE -ne 0) { throw 'Cannot access this folder from WSL.' }
    $linuxSource = ($linuxSource -replace "`0", '').Trim()
    if ($Setup) {
        Run-WSL -Arguments @('-d', $distro, '--', 'bash', '-lc', 'sudo apt-get update && sudo apt-get install -y curl ca-certificates')
        # Transfer source only; datasets and environments stay in Linux storage.
        Run-WSL -Arguments @('-d', $distro, '--', 'bash', '-c', 'set -euo pipefail; mkdir -p "$HOME/PollenStudio"; cd "$1"; tar --exclude=./.venv --exclude=./.tools --exclude=./.python --exclude=./studio_data --exclude=./models --exclude=./captures --exclude=./.git --exclude=./.env --exclude="*.zip" -cf - . | tar -xf - -C "$HOME/PollenStudio"; cd "$HOME/PollenStudio"; bash Setup-Linux.sh', 'pollen-setup', $linuxSource)
        Write-Host 'Setup complete. Double-click Start-Windows.cmd.'
    } else {
        Write-Host 'Open http://localhost:8765 in your Windows browser. Keep this terminal open.'
        Run-WSL -Arguments @('-d', $distro, '--', 'bash', '-lc', 'cd "$HOME/PollenStudio" && bash Start-Linux.sh')
    }
} catch {
    Write-Host $_ -ForegroundColor Red
    exit 1
}
