# ntu-msds-sd6104

- [ntu-msds-sd6104](#ntu-msds-sd6104)
  - [Installation](#installation)
    - [🖼️ WSL](#️-wsl)
    - [🚀 Quickstart with uv](#-quickstart-with-uv)
    - [🧪 Set Up the Environment](#-set-up-the-environment)
    - [✅ Activating the Environment](#-activating-the-environment)
  - [Usage](#usage)
    - [🧰 CLI Usage](#-cli-usage)
      - [🏃‍♂️ Basic Command](#️-basic-command)
    - [🎛 Available Flags \& Options](#-available-flags--options)
    - [📌 Example Usages](#-example-usages)

## Installation

### 🖼️ WSL

For windows users, follow the guide to install linux on windows using [WSL](https://learn.microsoft.com/en-us/windows/wsl/install). [Desborante-Core](https://github.com/Desbordante/desbordante-core) library is used for calculating FDs and INDs, which has been tested on Ubuntu and MacOS. To install this library on windows, WSL is needed. 



### 🚀 Quickstart with uv

This project uses uv to manage Python dependencies and environments. It's super fast and reproducible via lockfiles.

### 🧪 Set Up the Environment
Just run:
```
uv sync
```

### ✅ Activating the Environment
If uv doesn’t auto-activate it for you (e.g., if you’re not using direnv), you can activate it manually:

- macOS/Linux:
```
source .venv/bin/activate
```

- Windows (PowerShell):
```
.venv\Scripts\Activate
```

## Usage
### 🧰 CLI Usage

This project includes a CLI script, `main.py`, which supports several data processing and profiling features. Use the following flags to control what operations are performed.

#### 🏃‍♂️ Basic Command

To run the main program:

```bash
uv run python main.py [options]
```
Or, if you've activated your virtual environment manually:
```
python main.py [options]
```

### 🎛 Available Flags & Options
| Short Flag | Long Option           | Type   | Description                                                                 |
|------------|------------------------|--------|-----------------------------------------------------------------------------|
| -np        | --no_preprocess        | flag   | Disable preprocessing on the dataset *(enabled by default)*.                |
| -s         | --single_profile       | flag   | Perform single-column profiling.                                            |
| -rm        | --rule_mining          | flag   | Run association rule mining.                                                |
| -fd        | --func_dependencies    | string | Functional dependency discovery. Options: default, approximate, all.        |
| -ind       | --ind_dependencies     | string | Inclusion dependency discovery. Options: default, approximate, all.         |



### 📌 Example Usages
Run preprocessing and single-column profiling:
```
python main.py -p -s
```
Run association rule mining and functional dependencies (default method):
```
python main.py -rm -fd default
```
Run everything using approximate techniques:
```
python main.py -fd approximate -ind approximate -s -rm
```
Run with all discovery techniques:
```
python main.py -fd all -ind all
```

