# Setup: Jetson Orin Nano

How to reach the **Jetson Orin Nano Developer Kit** from a Windows PC and prepare the Python 3.9 environment for
the scripts in this folder. How to use the scripts afterwards: [README.md](README.md).

| Step | What | Why |
|---|---|---|
| [1](#1-first-connection-usb-cable--putty) | USB cable + PuTTY (serial console) | first login, without monitor, keyboard or network |
| [2](#2-ssh-over-the-usb-cable) | SSH over the same USB cable | a proper terminal: copy/paste, several sessions, file copy |
| [3](#3-wi-fi-from-the-terminal) | Wi-Fi from the terminal | network access, and SSH without the cable |
| [4](#4-remote-access-from-anywhere-tailscale) | Tailscale (background service) | SSH to the Jetson from any network |
| [5](#5-clock-and-time-zone) | Clock and time zone | the model uses the local time of day |
| [6](#6-get-the-code) | Get the code | this repository on the Jetson |
| [7](#7-python-39-environment) | Python 3.9 environment: venv **or** conda | the packages the scripts need |
| [8](#8-check-everything-works) | Check | run the scripts once |

**You need:** the Jetson Orin Nano Developer Kit with JetPack already flashed (microSD or NVMe), its power supply,
a **USB-C data cable** (charge-only cables do not work), and a Windows PC with [PuTTY](https://www.putty.org/).
Steps 1 and 2 use only the cable; from step 3 on, the Jetson also needs the Wi-Fi network the Shelly is on.

---

## 1. First connection: USB cable + PuTTY

The Jetson's USB-C port works as a **serial console** when it is connected to a PC: a text terminal on the Jetson,
with no network needed.

1. Power the Jetson on and connect its **USB-C port** to the PC. Wait about a minute for it to boot.
2. On the PC, open **Device Manager → Ports (COM & LPT)**. A new **USB Serial Device (COM*n*)** appears, e.g. `COM5`.
   Unplug and replug the cable if you are not sure which one it is.
3. In **PuTTY**:
   - **Session:** Connection type **Serial**, Serial line **`COM5`** (yours), Speed **`115200`**.
   - **Connection → Serial:** Data bits `8`, Stop bits `1`, Parity `None`, Flow control `None`.
   - Back in **Session**, type `jetson-usb-serial` under *Saved Sessions* and click **Save**, then **Open**.
4. Press **Enter** in the black window: a login prompt or the setup wizard appears.
   - **First boot without a monitor:** the setup wizard (licence, language, keyboard, **time zone**, user name and
     password) runs here in text mode. Pick the time zone of the umbrella (`Asia/Seoul` for the training data), and
     remember the user name and password. The Jetson may reboot at the end; press Enter again to log in.
   - **Later:** log in with that user name and password.
5. Check what is installed:
   ```bash
   uname -m                    # aarch64
   cat /etc/nv_tegra_release   # L4T release, e.g. R36 = JetPack 6
   lsb_release -ds             # Ubuntu version: 22.04 for JetPack 6, 20.04 for JetPack 5
   ```

## 2. SSH over the USB cable

The same cable also creates a small network between the PC and the Jetson: the **Jetson is `192.168.55.1`**
(the PC gets `192.168.55.100` automatically). SSH is a better terminal than the serial console: copy/paste works,
you can open several sessions, and files can be copied with `scp`.

- **PuTTY:** Session → Host Name **`192.168.55.1`**, Port **`22`**, Connection type **SSH**.
  Save it as `jetson-usb-ssh`, then **Open** and log in.
- **Or** the built-in OpenSSH in Windows Terminal / PowerShell:
  ```powershell
  ssh <user>@192.168.55.1
  ```

If the connection is refused, start the SSH server from the serial console (step 1):

```bash
sudo systemctl enable --now ssh
```

## 3. Wi-Fi from the terminal

Over SSH (step 2) or the serial console, connect the Jetson to the **same Wi-Fi network as the Shelly** with
NetworkManager (`nmcli`):

```bash
nmcli radio wifi on                              # make sure Wi-Fi is switched on
nmcli device status                              # wlan0 should be listed
sudo nmcli device wifi rescan
nmcli device wifi list                           # visible networks, signal strength
sudo nmcli --ask device wifi connect "YourWiFi"  # asks for the password (it is not saved in the shell history)
```

For a hidden network, add `hidden yes` to the connect command. Then check:

```bash
nmcli connection show --active                   # "YourWiFi" on wlan0
ip -4 addr show wlan0                            # the Jetson's Wi-Fi address, e.g. inet 192.168.1.42/24
ping -c 3 8.8.8.8                                # internet works
ping -c 3 <shelly-ip>                            # the Shelly is reachable
```

The connection is saved and comes back after a reboot (check: `nmcli -f NAME,AUTOCONNECT connection show`).

**Give the Jetson a fixed address**, so SSH and scripts always find it. The easiest way is a DHCP reservation for
the Jetson in the router. Otherwise set it on the Jetson; do this over the **USB** connection, because the
Wi-Fi connection drops while the address changes:

```bash
sudo nmcli connection modify "YourWiFi" ipv4.method manual \
    ipv4.addresses 192.168.1.42/24 ipv4.gateway 192.168.1.1 ipv4.dns "192.168.1.1 8.8.8.8"
sudo nmcli connection up "YourWiFi"
```

(Use an unused address of your network; the gateway is usually the router.)

From now on you can unplug the USB cable and connect over Wi-Fi: `ssh <user>@192.168.1.42`
(in PuTTY: Host Name `192.168.1.42`, saved as `jetson-wifi`).

**If `wlan0` is missing or Wi-Fi is "unavailable":** `rfkill list` shows whether it is blocked;
`sudo rfkill unblock wifi` unblocks it.

## 4. Remote access from anywhere: Tailscale

SSH over Wi-Fi (step 3) only works from the same network. [Tailscale](https://tailscale.com) gives the Jetson a
private address you can reach from your PC anywhere (another Wi-Fi, mobile data), with no router or port settings.
It runs as a **background service** (`tailscaled`): once logged in, it stays connected after you close SSH and
reconnects by itself after every reboot.

You need a free Tailscale account, and Tailscale installed and logged in on your **PC** too
([download](https://tailscale.com/download)), with the same account.

1. **Install** with Tailscale's install script (it picks the arm64 package for the Jetson and adds Tailscale's
   package repository, so `sudo apt upgrade` keeps it up to date):
   ```bash
   curl -fsSL https://tailscale.com/install.sh | sh
   ```
2. **Make sure the background service runs now and starts at every boot:**
   ```bash
   sudo systemctl enable --now tailscaled
   systemctl is-enabled tailscaled     # enabled
   systemctl is-active tailscaled      # active
   ```
   Tailscale needs the `tun` kernel module. If the service fails to start, load it now and at every boot:
   ```bash
   sudo modprobe tun
   echo tun | sudo tee /etc/modules-load.d/tun.conf
   ```
3. **Connect the Jetson to your Tailscale network**, choosing **one** of:
   - **Login link** (simplest): run
     ```bash
     sudo tailscale up
     ```
     It prints a link such as `https://login.tailscale.com/a/...`. Open it on your PC or phone and log in; the command
     then finishes by itself. The Jetson does not need a browser.
   - **Auth key** (nothing to click, good for scripts): create a key in the Tailscale admin console
     (**Settings → Keys → Generate auth key**), then
     ```bash
     sudo tailscale up --auth-key=tskey-auth-XXXXXXXX
     ```
     Treat the key like a password: do not commit it to git or leave it in shared notes.

   This is needed only once: the login is saved and the service reconnects on its own after reboots or
   network changes. `tailscale up` itself does not need to stay running; the connection lives in `tailscaled`.
4. **Give it a recognisable name** and check it:
   ```bash
   sudo tailscale set --hostname=umbrella-jetson
   tailscale status                    # the Jetson and your PC, both listed
   tailscale ip -4                     # the Jetson's Tailscale address, e.g. 100.x.y.z
   ```
5. **Keep it connected for good:** in the admin console, **Machines → umbrella-jetson → ⋯ → Disable key expiry**.
   Otherwise the login expires after some months and the Jetson silently drops off Tailscale.
   (If that happened: `sudo tailscale up --force-reauth`.)
6. **Connect from the PC** (with Tailscale running there), from any network:
   ```powershell
   ssh <user>@umbrella-jetson           # or ssh <user>@100.x.y.z
   ```
   In PuTTY: Host Name `umbrella-jetson` (or the `100.x.y.z` address), saved as `jetson-tailscale`.
   This is normal SSH through Tailscale's private network, so the Jetson's own SSH server and password are used.

Useful commands:

| Command | What it does |
|---|---|
| `tailscale status` | devices on your Tailscale network and how they are connected |
| `tailscale ip -4` | the Jetson's Tailscale address |
| `sudo tailscale down` | disconnect for now (the service keeps running); `sudo tailscale up` reconnects |
| `sudo tailscale logout` | disconnect and log out; the next `tailscale up` asks to log in again |
| `systemctl status tailscaled` | whether the background service is running |
| `journalctl -u tailscaled -n 50` | the service's last log lines, if something is wrong |

Tailscale only carries the remote SSH session. The Jetson still talks to the Shelly over the local Wi-Fi, so
the Shelly does not need Tailscale.

## 5. Clock and time zone

The model uses the **local time of day** (heat only counts in the daytime), and the training data is Korean time.
Check that the clock is right and synchronised:

```bash
timedatectl                                   # "Time zone: Asia/Seoul", "System clock synchronized: yes"
sudo timedatectl set-timezone Asia/Seoul      # if the time zone is different
```

The clock synchronises over the internet (step 3).

## 6. Get the code

```bash
sudo apt update
sudo apt install -y git
git clone https://github.com/Sekigahara/prototype-automatic-umbrella.git ~/prototype-automatic-umbrella
cd ~/prototype-automatic-umbrella
```

Alternatively, copy it from the PC (PowerShell, from the folder that contains the project):

```powershell
scp -r prototype-automatic-umbrella <user>@192.168.1.42:~/
```

`run_simulation.py` and `test_shelly.py` only need `deployment/`. `extract_evaluation.py` also needs the training code
(`modules/`) and `labeled_dataset/`, which come with the repository. `dataset/` (raw KMA files) is git-ignored; copy it
over separately only if you want to re-run `simulation.ipynb`.

## 7. Python 3.9 environment

The scripts need **Python 3.9**. JetPack 6 (Ubuntu 22.04) ships 3.10 and JetPack 5 (Ubuntu 20.04) ships 3.8,
so 3.9 is installed next to it. Use **either** a venv (option A) **or** conda (option B), not both.

**What to install:** `requirements.txt` pins everything the whole project uses, including the training packages
(hmmlearn, scikit-learn, scipy, matplotlib). The scripts on the Jetson need only part of it. If the full install
fails on aarch64, install just the device packages instead (same versions):

```bash
python -m pip install numpy==1.26.4 pandas==2.3.3 pythermalcomfort==3.3.0 numba==0.60.0 requests==2.32.5 joblib==1.5.3
```

### Option A: venv

1. **Install Python 3.9.**
   - JetPack 6 / Ubuntu 22.04, from the deadsnakes PPA:
     ```bash
     sudo apt update
     sudo apt install -y software-properties-common
     sudo add-apt-repository -y ppa:deadsnakes/ppa
     sudo apt install -y python3.9 python3.9-venv
     ```
   - JetPack 5 / Ubuntu 20.04: Python 3.9 is in Ubuntu's own repositories:
     ```bash
     sudo apt update
     sudo apt install -y python3.9 python3.9-venv
     ```

   If `apt` cannot find `python3.9`, use option B.
2. **Create and activate the environment**, from the repository root:
   ```bash
   cd ~/prototype-automatic-umbrella
   python3.9 -m venv .venv
   source .venv/bin/activate
   which python         # ~/prototype-automatic-umbrella/.venv/bin/python
   python --version     # Python 3.9.x
   ```
3. **Install the packages:**
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r deployment/requirements.txt
   python -m pip check  # "No broken requirements found."
   ```
4. **Optional, for the notebooks:** `python -m ipykernel install --user --name umbrella-venv --display-name "Python 3.9 (umbrella venv)"`

Activate it in every new shell: `source ~/prototype-automatic-umbrella/.venv/bin/activate`.
To start over: `deactivate`, then `rm -rf ~/prototype-automatic-umbrella/.venv`.

### Option B: Miniconda (own environment)

Conda installs Python 3.9 itself, so no system Python or PPA is needed. Everything goes into its own environment,
here called **`umbrella`**; `base` is never touched. Use any name except `base`, and the same name in every step.
The installer steps follow Anaconda's
[Miniconda Linux install guide](https://www.anaconda.com/docs/getting-started/miniconda/install/linux-install)
for **aarch64** (ARM64, the Jetson). On an x86_64 PC, replace `aarch64` with `x86_64` in steps 2–4.

1. **Check the architecture** and that conda is not installed yet:
   ```bash
   uname -m             # aarch64 on the Jetson
   conda --version      # "command not found" -> install it (steps 2-4); a version -> skip to step 5
   ```
   Do not install a second copy over an existing one: if `~/miniconda3` already exists, the installer stops with
   "File or directory already exists". Never force it with `-u` or `-f`; that can break the installation.
2. **Download the aarch64 installer** into your home folder:
   ```bash
   cd ~
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
   ```
3. **Verify it** (recommended): compare the hash with the one listed for `Miniconda3-latest-Linux-aarch64.sh`
   at <https://repo.anaconda.com/miniconda/>. Only continue if they match.
   ```bash
   sha256sum ~/Miniconda3-latest-Linux-aarch64.sh
   ```
4. **Install:**
   ```bash
   bash ~/Miniconda3-latest-Linux-aarch64.sh
   ```
   Answer the prompts in this order:
   - press **Return** to read the licence, then type **`yes`** to accept it;
   - press **Return** to install to the default `/home/<user>/miniconda3`;
   - type **`yes`** to initialize conda, so the `conda` command works in new shells.

   It ends with "Thank you for installing Miniconda3!". Then reload the shell and check:
   ```bash
   source ~/.bashrc     # or close and reopen the terminal
   conda list           # "# packages in environment at /home/<user>/miniconda3:" and a package list
   ```
   If `conda` is still not found: run `~/miniconda3/bin/conda init`, then `source ~/.bashrc` again.
5. **Keep `base` out of the way.** After installing, every new shell starts in `base` (the prompt shows `(base)`),
   so a stray `pip install` would land there. Turn that off; `conda` keeps working, and `conda activate` still
   enters `base` when you really want it:
   ```bash
   conda config --set auto_activate_base false
   conda config --describe auto_activate_base    # shows: auto_activate_base: false
   source ~/.bashrc                               # the prompt no longer starts with (base)
   ```
6. **Create the environment with Python 3.9** (change `python=3.9` here for another version).
   It uses the conda-forge channel only (`--override-channels`), which has Python 3.9 builds for aarch64:
   ```bash
   conda create -n umbrella -c conda-forge --override-channels python=3.9 pip -y
   ```
7. **Isolate it from user-level packages.** Packages in `~/.local` (from `pip install --user`) would otherwise
   leak into the environment and mix versions:
   ```bash
   conda env config vars set -n umbrella PYTHONNOUSERSITE=1
   ```
8. **Activate it and check you are really inside it** before installing anything:
   ```bash
   conda activate umbrella
   which python         # must be ~/miniconda3/envs/umbrella/bin/python, not ~/miniconda3/bin/python (base)
   python --version     # Python 3.9.x
   python -c "import platform; print(platform.machine())"   # aarch64
   ```
9. **Install the requirements with the environment's own pip**, from the repository root.
   `python -m pip` (not plain `pip`) guarantees it installs into `umbrella`:
   ```bash
   cd ~/prototype-automatic-umbrella
   python -m pip install --upgrade pip
   python -m pip install -r deployment/requirements.txt
   python -m pip check                        # "No broken requirements found."
   ```
10. **Optional, for the notebooks:** register it as a Jupyter kernel.
    ```bash
    python -m ipykernel install --user --name umbrella --display-name "Python 3.9 (umbrella)"
    ```

Run `conda activate umbrella` in every new shell before using the scripts.

Rules that keep it clean:
- Install packages only with `python -m pip` inside the activated environment. Do not also `conda install` the
  same packages, or conda and pip will each manage their own copy.
- If `pip` finds no prebuilt aarch64 wheel for a package and tries to compile it, install that one package from
  conda-forge first and rerun step 9:
  `conda install -n umbrella -c conda-forge --override-channels <package>=<version from requirements.txt>`.
- Keep `-c conda-forge --override-channels` on every `conda create` / `conda install`, as above, so the
  environment only uses conda-forge packages.
- Start over at any time, without affecting `base` or other environments:
  ```bash
  conda deactivate
  conda env remove -n umbrella
  ```
- Check which environments exist and which one is active (marked `*`): `conda env list`.

Miniconda on aarch64, from Anaconda's guide:
- The `linux-aarch64` builds target server-class ARM CPUs (Neoverse N1/N2) and may not run on some boards,
  Raspberry Pi in particular. If `conda` or `python` stops with "Illegal instruction", Miniconda does not support
  that CPU: use option A (venv) instead.
- Behind a proxy or firewall, `https://repo.anaconda.com` and `https://anaconda.org` must be reachable for the
  installer and for conda; `https://conda.anaconda.org` is needed for the conda-forge packages in step 6.
- A broken Miniconda can be repaired without losing environments: rename `~/miniconda3` to `~/miniconda_old`,
  run the installer again into `~/miniconda3`, copy the old contents back with
  `rsync -a ~/miniconda_old/ ~/miniconda3/`, check with `conda info --envs`, then delete `~/miniconda_old`.

## 8. Check everything works

With the environment active:

```bash
cd ~/prototype-automatic-umbrella/deployment
python run_simulation.py --limit 1 --every 60      # replays one clip through the model
python test_shelly.py --host <shelly-ip>           # read-only: connection and status, never moves the motor
```

Next: [README.md](README.md) for everything the scripts can do, and the safe way to test the motor.

## Troubleshooting

| Problem | Fix |
|---|---|
| No new COM port in Device Manager | Use a USB-C **data** cable, wait for the Jetson to finish booting, try another USB port on the PC |
| PuTTY window stays empty | Press Enter; check Speed `115200` and the COM number |
| `ssh` to `192.168.55.1` refused or times out | Over the serial console: `sudo systemctl enable --now ssh`; check the cable is still in |
| Wi-Fi network not in `nmcli device wifi list` | `sudo nmcli device wifi rescan`, wait a few seconds and list again; check the network name and that the Jetson is in range |
| Lost the Jetson's Wi-Fi address | Connect over USB again (step 2) and run `ip -4 addr show wlan0` |
| `python3.9: command not found` | The environment is not active, or Python 3.9 is not installed (step 7) |
| `conda: command not found` | `~/miniconda3/bin/conda init`, then `source ~/.bashrc` |
| `ssh umbrella-jetson` fails from outside | Tailscale running and logged in on the PC too; on the Jetson `tailscale status` lists it; key expiry (step 4.5) |
| Scripts cannot reach the Shelly | Jetson and Shelly must be on the same network: `ping <shelly-ip>` from the Jetson |
