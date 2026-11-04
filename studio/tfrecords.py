"""Small dependency-free TF Example reader for importing existing detections."""
import hashlib
import json
import struct
import zipfile
from pathlib import Path
from studio.core import DATA, connect, validate_boxes


def varint(data, pos):
    value = 0
    for shift in range(0, 70, 7):
        byte = data[pos]; pos += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, pos
    raise ValueError('Invalid protobuf integer')


def fields(data):
    pos = 0
    while pos < len(data):
        tag, pos = varint(data, pos)
        wire = tag & 7
        if wire == 2:
            size, pos = varint(data, pos)
            value = data[pos:pos+size]; pos += size
        elif wire == 0:
            value, pos = varint(data, pos)
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
            value = data[pos:pos+size]; pos += size
        else:
            raise ValueError('Unsupported protobuf wire type')
        if pos > len(data):
            raise ValueError('Truncated protobuf')
        yield tag >> 3, value


def example(data):
    features = dict(fields(data))[1]
    result = {}
    for _, entry in fields(features):
        pair = dict(fields(entry)); key = pair[1].decode()
        kind, packed = next(fields(pair[2]))
        vals = [v for _, v in fields(packed)]
        if kind == 2:
            vals = list(struct.unpack('<' + 'f' * (len(vals[0])//4), vals[0])) if vals else []
        elif kind == 3:
            raw = vals[0] if vals else b''; vals = []; pos = 0
            while pos < len(raw):
                v, pos = varint(raw, pos); vals.append(v)
        result[key] = vals
    return result


def import_archive(path):
    count = 0
    with zipfile.ZipFile(Path(path).expanduser()) as archive, connect() as db:
        manifest = {}
        if 'manifest.json' in archive.namelist():
            manifest = {r['filename']: r for r in json.loads(archive.read('manifest.json'))['records']}
        for name in archive.namelist():
            if not name.endswith('.tfrecord'):
                continue
            split = 'val' if any(part.lower() in ('valid', 'val', 'validation') for part in Path(name).parts[:-1]) else 'train'
            with archive.open(name) as stream:
                while header := stream.read(12):
                    if len(header) != 12:
                        raise ValueError('Truncated TFRecord header')
                    size = struct.unpack('<Q', header[:8])[0]
                    if size > 100_000_000:
                        raise ValueError('Record is too large')
                    raw = stream.read(size)
                    if len(raw) != size or len(stream.read(4)) != 4:
                        raise ValueError('Truncated TFRecord')
                    f = example(raw)
                    encoded = f['image/encoded'][0]
                    digest = hashlib.sha256(encoded).hexdigest()
                    filename = f['image/filename'][0].decode()
                    if filename in manifest and digest != manifest[filename]['sha256']:
                        raise ValueError('Image does not match source manifest')
                    if db.execute('SELECT 1 FROM purged_images WHERE id=?',(digest[:24],)).fetchone():
                        continue
                    classes = f.get('image/object/class/label', [])
                    if any(c not in (1,2,3) for c in classes) or f.get('image/object/class/text', []) != [ ('pollen','blue_nectar','red_nectar')[c-1].encode() for c in classes]:
                        raise ValueError('Expected TensorFlow class 1 (pollen)')
                    coords = [f.get('image/object/bbox/'+k, []) for k in ('xmin','ymin','xmax','ymax')]
                    if any(len(c) != len(classes) for c in coords):
                        raise ValueError('Mismatched box features')
                    boxes = validate_boxes([dict(zip(('x1','y1','x2','y2'), v)) | {'class_id':c-1} for v,c in zip(zip(*coords),classes)])
                    suffix = '.png' if encoded.startswith(b'\x89PNG') else '.jpg'
                    target = DATA / 'original' / split / (digest + suffix)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(encoded)
                    prior = db.execute('SELECT split FROM images WHERE id=?', (digest[:24],)).fetchone()
                    if prior and prior['split'] != split:
                        raise ValueError('Identical image occurs in train and validation.')
                    count += db.execute('INSERT OR IGNORE INTO images(id,path,session,split,status,boxes) VALUES (?,?,?,?,?,?)', (digest[:24], str(target), 'Original dataset / '+split, split, 'include', json.dumps(boxes))).rowcount
                    if count % 100 == 0:
                        print(f'Imported {count} images', flush=True)
    return count


def encode_varint(value):
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128); value >>= 7
    result.append(value)
    return bytes(result)


def message(field, value):
    return encode_varint((field << 3) | 2) + encode_varint(len(value)) + value


def feature(value, kind):
    if kind == 'bytes':
        return message(1, b''.join(message(1, v) for v in value))
    if kind == 'float':
        return message(2, message(1, struct.pack('<'+'f'*len(value), *value)))
    return message(3, message(1, b''.join(encode_varint(v) for v in value)))


def encode_example(features):
    entries = []
    for key, (kind, values) in features.items():
        entries.append(message(1, message(1, key.encode()) + message(2, feature(values, kind))))
    return message(1, b''.join(entries))


_CRC_TABLE = []
for _i in range(256):
    _crc = _i
    for _ in range(8):
        _crc = (_crc >> 1) ^ (0x82F63B78 if _crc & 1 else 0)
    _CRC_TABLE.append(_crc)


def crc32c(data):
    try:
        import google_crc32c
        return google_crc32c.value(data)
    except ImportError:
        crc = 0xFFFFFFFF
        for byte in data:
            crc = _CRC_TABLE[(crc ^ byte) & 255] ^ (crc >> 8)
        return crc ^ 0xFFFFFFFF


def masked_crc(data):
    crc = crc32c(data)
    return struct.pack('<I', (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF)


def image_size(encoded):
    if encoded.startswith(b'\x89PNG\r\n\x1a\n'):
        return (*struct.unpack('>II', encoded[16:24]), b'png')
    if encoded[:2] == b'\xff\xd8':
        pos = 2
        while pos < len(encoded):
            if encoded[pos] != 255:
                raise ValueError('Invalid JPEG marker')
            while encoded[pos] == 255: pos += 1
            marker = encoded[pos]; pos += 1
            if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7: continue
            length = struct.unpack('>H', encoded[pos:pos+2])[0]
            if marker in (0xC0,0xC1,0xC2,0xC3,0xC5,0xC6,0xC7,0xC9,0xCA,0xCB,0xCD,0xCE,0xCF):
                height, width = struct.unpack('>HH', encoded[pos+3:pos+7])
                return width, height, b'jpeg'
            if length < 2: raise ValueError('Invalid JPEG segment')
            pos += length
    raise ValueError('TFRecord export supports JPEG and PNG. Convert WebP images before importing.')


def export_records(rows, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    from studio.classes import CLASSES
    label_map = ''.join("item {\n  id: %d\n  name: '%s'\n}\n" % (i+1,name) for i,name in enumerate(CLASSES))
    (directory / 'label_map.pbtxt').write_text(label_map)
    for split, folder in [('train','train'), ('val','valid')]:
        target = directory / folder
        target.mkdir(exist_ok=True)
        (target / 'pollen_label_map.pbtxt').write_text(label_map)
        selected = [r for r in rows if r['split'] == split]
        with (target / 'pollen.tfrecord').open('wb') as stream:
            for i, row in enumerate(selected):
                encoded = Path(row['path']).read_bytes()
                width, height, fmt = image_size(encoded)
                boxes = validate_boxes(row['boxes'])
                f = {'image/height':('int',[height]), 'image/width':('int',[width]),
                     'image/filename':('bytes',[Path(row['path']).name.encode()]),
                     'image/source_id':('bytes',[row['id'].encode()]),
                     'image/key/sha256':('bytes',[hashlib.sha256(encoded).hexdigest().encode()]),
                     'image/encoded':('bytes',[encoded]), 'image/format':('bytes',[fmt]),
                     'image/object/class/text':('bytes',[CLASSES[b.get('class_id',0)].encode() for b in boxes]),
                     'image/object/class/label':('int',[b.get('class_id',0)+1 for b in boxes])}
                for feature_name, coord in [('xmin','x1'),('ymin','y1'),('xmax','x2'),('ymax','y2')]:
                    f['image/object/bbox/'+feature_name] = ('float',[b[coord] for b in boxes])
                record = encode_example(f)
                length = struct.pack('<Q',len(record))
                stream.write(length+masked_crc(length)+record+masked_crc(record))
                if i % 25 == 0: print(f'Exporting {split}: {i+1}/{len(selected)}', flush=True)
