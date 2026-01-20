# Windows EXE build instructions (PyInstaller)

**Most users do NOT need to build anything.**  
Download the pre-built Windows executable from the Releases page:
- https://github.com/milkywaycj/maxdiff/releases/latest

This file exists for transparency and reproducibility: it documents the exact command used to produce the Windows EXE from the Python source code in this repository.

---

## Who is this for?
These steps are only for people who want to rebuild the EXE themselves (for example: to verify the build, modify the code, or build on another machine).

---

## Requirements (for rebuilding)
- Windows 10/11
- Python 3.x
- pip

---

## Rebuild steps (optional)

From the repository root:

### 1) Create and activate a virtual environment
```bat
python -m venv .venv
.venv\Scripts\activate
```
### 2) Install dependencies
```bat
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller
```
### 3) Build the EXE (exact command used)

If the Python file is in src/:
```bat
python -m PyInstaller --onefile --noconsole --clean --hidden-import=openpyxl --hidden-import=matplotlib.backends.backend_pdf --hidden-import=matplotlib.backends.backend_agg --hidden-import=matplotlib.backends.backend_tkagg --collect-submodules=matplotlib.backends --collect-submodules=matplotlib --collect-all=customtkinter src/MaxDiff_Data_Analyzer_v2-1.py
```
(If your file is not in src/, replace the last argument with MaxDiff_Data_Analyzer_v2-1.py.)
(The hidden-import and collect flags ensure matplotlib and CustomTkinter backends are bundled correctly.)

### Output

The executable will be created in the dist\ folder, for example:
dist\MaxDiff_Data_Analyzer_v2-1.exe

### Notes

--noconsole is used for GUI apps. Remove it if you want a console window.
If someone gets import errors, it usually means requirements.txt is missing a package.
This EXE is not code-signed, so Windows may show an “unknown publisher” / SmartScreen warning.

---
These instructions document the exact command used to build the published Windows executable.
