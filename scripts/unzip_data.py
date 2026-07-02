"""Extract data.zip with short filenames to avoid "File name too long" errors."""
import zipfile
import os
import shutil

src = '/root/dimsum/data/data.zip'
dst = '/root/dimsum/data/audio_data'
os.makedirs(dst, exist_ok=True)

mapping = []
total = 0

with zipfile.ZipFile(src, 'r') as z:
    # Count extractable files first
    for info in z.infolist():
        if '__MACOSX' in info.filename or info.filename.startswith('/') or info.is_dir():
            continue
        total += 1

    idx = 0
    for info in z.infolist():
        if '__MACOSX' in info.filename or info.filename.startswith('/') or info.is_dir():
            continue
        ext = os.path.splitext(info.filename)[1] or '.wav'
        short_name = f'audio_{idx:05d}{ext}'
        out_path = os.path.join(dst, short_name)
        with z.open(info) as srcf, open(out_path, 'wb') as dstf:
            shutil.copyfileobj(srcf, dstf)
        mapping.append((short_name, info.filename))
        idx += 1

print(f'Extracted {total} files')

with open(os.path.join(dst, 'mapping.csv'), 'w', encoding='utf-8') as f:
    f.write('short_name,original_path\n')
    for s, o in mapping:
        f.write(f'{s},{o}\n')
print('Mapping saved to mapping.csv')
