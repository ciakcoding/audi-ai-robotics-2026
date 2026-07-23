# Setup

Use Python 3.11 on Windows.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\verify_frozen_snapshot.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

For training with CUDA, install a CUDA-enabled PyTorch build appropriate for
the workstation if the default package does not provide one. The included
policy can always be evaluated on CPU.
