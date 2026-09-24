"""Minimal, fast FITS reader for single-HDU 2D images (what capture software writes)."""
import numpy as np

_DTYPES = {8: 'u1', 16: '>i2', 32: '>i4', -32: '>f4', -64: '>f8'}


def _parse_header(buf):
    hdr, off = {}, 0
    while off + 2880 <= len(buf):
        blk = buf[off:off + 2880]
        off += 2880
        for i in range(36):
            card = blk[i * 80:(i + 1) * 80].decode('ascii', 'replace')
            key = card[:8].strip()
            if key == 'END':
                return hdr, off
            if card[8:10] == '= ':
                val = card[10:]
                if val.lstrip().startswith("'"):
                    val = val.lstrip()[1:].split("'")[0].strip()
                else:
                    val = val.split('/')[0].strip()
                hdr[key] = val
    raise ValueError('FITS header END not found')


def read_header(path):
    """Read just the header (usually 1-2 blocks), return (hdr, data_offset)."""
    buf = b''
    with open(path, 'rb') as fh:
        while True:
            blk = fh.read(2880)
            if len(blk) < 2880:
                raise ValueError(f'truncated FITS header: {path}')
            buf += blk
            try:
                return _parse_header(buf)
            except ValueError:
                continue


def _to_float(a, hdr):
    img = a.astype(np.float32)
    bscale, bzero = float(hdr.get('BSCALE', 1)), float(hdr.get('BZERO', 0))
    if bscale != 1:
        img *= bscale
    if bzero:
        img += bzero
    return img


def read_fits(path):
    """Return (header, image). Unsigned 16-bit data (BITPIX=16, BZERO=32768: every
    CMOS camera) comes back as native uint16 without a float copy, to keep memory
    low; anything else comes back as float32 in physical units."""
    hdr, off = read_header(path)
    if int(hdr.get('NAXIS', 0)) < 2:
        raise ValueError(f'not a 2D image: {path}')
    w, h = int(hdr['NAXIS1']), int(hdr['NAXIS2'])
    bitpix = int(hdr['BITPIX'])
    with open(path, 'rb') as fh:
        fh.seek(off)
        a = np.fromfile(fh, dtype=_DTYPES[bitpix], count=w * h).reshape(h, w)
    if bitpix == 16 and float(hdr.get('BZERO', 0)) == 32768 and float(hdr.get('BSCALE', 1)) == 1:
        u = a.byteswap(inplace=True).view(np.uint16)   # in place: big-endian int16 -> native
        u ^= 0x8000                                     # +32768 == flip the sign bit
        return hdr, u
    return hdr, _to_float(a, hdr)


def read_crop(path, cx=0.5, cy=0.5, cw=2000, ch=1300):
    """Read only the rows needed for a full-resolution crop centred at (cx, cy) fractions."""
    hdr, off = read_header(path)
    w, h = int(hdr['NAXIS1']), int(hdr['NAXIS2'])
    dt = np.dtype(_DTYPES[int(hdr['BITPIX'])])
    cw, ch = min(cw, w), min(ch, h)
    x0 = int(np.clip(cx * w - cw / 2, 0, w - cw))
    y0 = int(np.clip(cy * h - ch / 2, 0, h - ch))
    with open(path, 'rb') as fh:
        fh.seek(off + y0 * w * dt.itemsize)
        a = np.frombuffer(fh.read(ch * w * dt.itemsize), dtype=dt).reshape(ch, w)[:, x0:x0 + cw]
    return _to_float(a, hdr)
