from PyInstaller.utils.hooks import collect_data_files, collect_submodules
import os
import platform

def collect_assets():
    assets_path = os.path.join(os.getcwd(), 'assets')
    assets = []
    for file in os.listdir(assets_path):
        file_path = os.path.join(assets_path, file)
        if os.path.isfile(file_path):
            assets.append(('assets/' + file, 'assets'))
    return assets

block_cipher = None

if platform.system() == 'Windows':
    exe_name = 'MetaVision-Windows.exe'
elif platform.system() == 'Linux':
    exe_name = 'MetaVision-Linux'
else:
    exe_name = 'MetaVision-Mac'
    
a = Analysis(
    ['main.py'],
    pathex=[SPECPATH],
    binaries=[],
    datas=collect_assets(),
    hiddenimports=[
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name=exe_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    codesign_identity=None,
    entitlements_file=None,
)
