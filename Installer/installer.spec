# -*- mode: python ; coding: utf-8 -*-

import os

ROOT = os.path.abspath(os.path.join(SPECPATH, '..'))

a = Analysis(
    ['installer.py'],
    pathex=[],
    binaries=[],
    datas=[
        # Embedded Python runtime (used to run the service on the target machine)
        ('python-embed', 'python-embed'),

        # Branding
        ('satellite.ico', '.'),
        ('logo.png', '.'),

        # Windows service scripts
        (os.path.join(ROOT, 'services', 'windows', 'dataGatheringService.py'),
         os.path.join('services', 'windows')),
        (os.path.join(ROOT, 'services', 'windows', 'dataGatheringServiceLogic.py'),
         os.path.join('services', 'windows')),

        # ML package
        (os.path.join(ROOT, 'ml'), 'ml'),

        # Dashboard package
        (os.path.join(ROOT, 'dashboard'), 'dashboard'),

        # ExoNet ONNX model
        (os.path.join(ROOT, 'exonet.onnx'), '.'),
        (os.path.join(ROOT, 'exonet.onnx.data'), '.'),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='installer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['satellite.ico'],
)
