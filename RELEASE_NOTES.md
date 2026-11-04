Pollen Studio is a local annotation and training workspace from Harambe Cartel, Team 20912.

Choose the package for your operating system:

- **Linux:** extract and run `bash Setup-Linux.sh`, then `bash Start-Linux.sh`.
- **Windows:** extract and double-click `Setup-Windows.cmd`, then `Start-Windows.cmd`. Training runs inside WSL2 Ubuntu 22.04. First-time installation may require administrator permission, an Ubuntu user account, and a Windows restart.

Open http://localhost:8765 after starting the server. Internet access is required for setup. NVIDIA training requires a compatible host driver; the training runtime is installed separately from the Training tab.

Both packages contain the same application source plus platform launchers. No models or datasets are included. Supports pollen, blue Nectar, and red Nectar, with photo uploads of up to 1,000 files / 512 MB.

Linux application tests and fresh-workspace checks were performed by the maintainers. A real Windows/RTX 3050 installation has not yet been tested. GPU training and Coral compilation remain dependent on external runtime downloads and compatible hardware.

MIT licensed. Retain the Harambe Cartel (Team 20912) copyright and license notice when redistributing.
