# PyInstaller spec: a single self-contained executable, no installer.
#
# One-file mode is deliberate. The audience is graduate students who should be able
# to download one file and double-click it, on a lab machine where they may not have
# permission to run an installer or to `pip install` anything.
#
# Build with:  pyinstaller packaging/microscopevuilder.spec --noconfirm

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# The component catalog is data, not code, so it has to be carried explicitly.
datas = collect_data_files("microscopevuilder", includes=["assets/*.toml"])

# Qt ships a great deal that a 2D scene graph never touches. Excluding it keeps the
# download to something a student will actually wait for.
excludes = [
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.Qt3DAnimation", "PySide6.Qt3DExtras",
    "PySide6.QtQuick", "PySide6.QtQuick3D", "PySide6.QtQml", "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets", "PySide6.QtCharts", "PySide6.QtDataVisualization",
    "PySide6.QtBluetooth", "PySide6.QtNetworkAuth", "PySide6.QtPositioning",
    "PySide6.QtRemoteObjects", "PySide6.QtSensors", "PySide6.QtSerialPort",
    "PySide6.QtSql", "PySide6.QtTest", "PySide6.QtWebSockets", "PySide6.QtWebChannel",
    "PySide6.QtPdf", "PySide6.QtPdfWidgets", "PySide6.QtDesigner", "PySide6.QtHelp",
    "PySide6.QtUiTools", "PySide6.QtSpatialAudio", "PySide6.QtTextToSpeech",
    "matplotlib", "scipy", "pandas", "PIL", "tkinter", "pytest", "setuptools",
    "IPython", "notebook", "sphinx",
]

a = Analysis(
    ["entry.py"],
    pathex=[str(Path(SPECPATH).parent)],
    binaries=[],
    datas=datas,
    hiddenimports=["microscopevuilder.ui.workspace"],
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="MicroscopeVuilder",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    # A windowed build on macOS and Windows; on Linux keep the console so a crash
    # is readable rather than silent.
    console=sys.platform.startswith("linux"),
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
