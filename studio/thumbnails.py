"""Small cached previews for the complete photo library."""
import hashlib
import json
from pathlib import Path
from studio.core import DATA


def thumbnail(row, boxes=True):
    from PIL import Image,ImageOps,ImageDraw
    source=Path(row['path']);stat=source.stat()
    labels=json.loads(row['boxes']) if isinstance(row['boxes'],str) else row['boxes']
    signature=hashlib.sha256((str(stat.st_mtime_ns)+str(stat.st_size)+json.dumps(labels if boxes else [])).encode()).hexdigest()[:16]
    destination=DATA/'thumbnails'/(row['id']+'-'+signature+'.jpg')
    if destination.is_file():return destination
    destination.parent.mkdir(parents=True,exist_ok=True)
    with Image.open(source) as image:
        image=ImageOps.contain(ImageOps.exif_transpose(image).convert('RGB'),(320,240))
        if boxes:
            draw=ImageDraw.Draw(image)
            for box in labels:
                draw.rectangle((box['x1']*image.width,box['y1']*image.height,box['x2']*image.width,box['y2']*image.height),outline='#b7f06d',width=2)
        preview=Image.new('RGB',(320,240),'#18211b')
        preview.paste(image,((320-image.width)//2,(240-image.height)//2))
        preview.save(destination,quality=82)
    return destination
