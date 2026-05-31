# Imagery Gaze Experiment

Eye-tracking experiment using PsychoPy and PyGaze (Gazepoint OpenGaze protocol).

---

## Setup on a new machine

### 1. Install Git
Download and install from https://git-scm.com/download/win  
Default options throughout the installer are fine.

### 2. Configure Git identity
Run these once after installing Git (required before any commit):
```powershell
git config --global user.email "you@example.com"
git config --global user.name "Your Name"
```

### 3. Clone the repository
```powershell
git clone https://github.com/daniilnet/imagery_gaze_experiment.git
cd imagery_gaze_experiment
```

> **Note:** Use the URL above — not the `/tree/main` branch URL from the GitHub browser.

### 4. Install Python 3.10.11
```powershell
Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe" -OutFile "python-3.10.11-amd64.exe"
.\python-3.10.11-amd64.exe
```
When the installer opens:
- Tick **"Add Python 3.10 to PATH"**
- Click **Install Now** (standard installation is fine)
- At the end, click **"Disable path length limit"** when prompted — this prevents issues with long package paths

### 5. Install uv
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Add uv to PATH permanently:
```powershell
[Environment]::SetEnvironmentVariable("Path", $env:Path + ";$env:USERPROFILE\.local\bin", "User")
```

Restart PowerShell, then verify it worked:
```powershell
uv --version
```

### 6. Create virtual environment and install all dependencies
```powershell
uv sync
```

This reads `pyproject.toml`, creates `.venv`, and installs PsychoPy, PyGaze, and all other dependencies automatically.

---

## Running the experiment
```powershell
uv run main.py
```

You will be prompted to enter a subject number in the terminal before the experiment window opens.

---

## Troubleshooting

**`uv` not recognised after install**  
Close and reopen PowerShell after setting the PATH, then retry.

**`pyproject.toml` TOML parse error on `python = ...`**  
This field was removed in newer versions of uv. Make sure you have the latest version of the repo (`git pull`) — the fix is already committed.

**Git not recognised in PyCharm terminal**  
Go to **File → Settings → Version Control → Git** and set the path to `C:\Program Files\Git\bin\git.exe`.

---

## Updating the repository

After making changes on any machine:
```powershell
git add -A
git commit -m "describe your changes"
git push
```

To pull the latest version on another machine:
```powershell
git pull
```
