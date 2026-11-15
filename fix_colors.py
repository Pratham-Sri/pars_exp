"""Batch replace stale color references in all HTML templates with monochrome equivalents."""
import glob
import re

replacements = [
    # CSS variable refs that no longer exist
    ('var(--green-dark)', '#000000'),
    ('var(--green)', '#000000'),
    ('var(--line)', '#000000'),
    # Specific color codes -> monochrome
    ('color:#16a34a', 'color:#000'),
    ('color:#1d4ed8', 'color:#000'),
    ('color:#1e3a8a', 'color:#000'),
    ('color:#b45309', 'color:#444'),
    ('color:#b91c1c', 'color:#444'),
    ('color:#991b1b', 'color:#000'),
    ('color:#dc2626', 'color:#444'),
    ('color:#15803d', 'color:#000'),
    ('color:#164e3d', 'color:#000'),
    ('color:#475569', 'color:#444'),
    ('color:#1e293b', 'color:#000'),
    ('color:#1e352b', 'color:#000'),
    ('color:#334155', 'color:#444'),
    ('color:#64748b', 'color:#888'),
    # Backgrounds -> monochrome grays
    ('background:#eef4ee', 'background:#f5f5f5'),
    ('background:#f8faf7', 'background:#f5f5f5'),
    ('background:#f1f5f0', 'background:#f5f5f5'),
    ('background:#f0fdf4', 'background:#f5f5f5'),
    ('background:#fff1f2', 'background:#f5f5f5'),
    ('background:#fef2f2', 'background:#f5f5f5'),
    ('background:#f0f6f2', 'background:#f5f5f5'),
    ('background:#edf3ee', 'background:#f5f5f5'),
    ('background:#f8fafc', 'background:#f5f5f5'),
    # Borders -> monochrome
    ('border:1px solid #dce5dc', 'border:1px solid #000'),
    ('border:1px solid #e0e6df', 'border:1px solid #000'),
    ('border:1px solid #e2e8f0', 'border:1px solid #000'),
    ('border-color:#fca5a5', 'border-color:#000'),
    ('border-bottom:2px solid #cbd7ce', 'border-bottom:2px solid #000'),
    ('border-bottom:1px solid #e5ede7', 'border-bottom:1px solid #ccc'),
    ('border-bottom:1px solid #e2e8e0', 'border-bottom:1px solid #ccc'),
    ('border-bottom:1px solid #eef2ec', 'border-bottom:1px solid #ccc'),
    ('border-bottom:1px solid #e5e7eb', 'border-bottom:1px solid #ccc'),
    ('border-top:1px dashed #d5e0d7', 'border-top:1px dashed #ccc'),
    ('border-left-color:#164e3d', 'border-left-color:#000'),
    ('border:1px solid #cbd5e1', 'border:1px solid #000'),
    ('border:1px solid #cbd7ce', 'border:1px solid #000'),
]

files = glob.glob(r'c:\Users\Pratham\PRICKA\dataset_lab\templates\*.html')
for fpath in files:
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    changed = False
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            changed = True
    
    if changed:
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed: {fpath.split(chr(92))[-1]}")
    else:
        print(f"Clean: {fpath.split(chr(92))[-1]}")
