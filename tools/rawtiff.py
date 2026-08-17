"""无压缩 TIFF 的最小依赖读取器，只依赖 numpy。

本数据集的影像为 512x512x224 uint16、无压缩、pixel-interleaved、RowsPerStrip=1，
标签为 512x512 int32。绝大多数文件的 strip 在文件里按行号升序连续存放，可以直接
memmap；但有少量文件的 strip 顺序是循环错位的，必须按 StripOffsets 重排行，
否则读出来的影像相对标签是竖向 roll 过的。
"""

import struct

import numpy as np

TAG_IMAGE_WIDTH = 256
TAG_IMAGE_LENGTH = 257
TAG_BITS_PER_SAMPLE = 258
TAG_COMPRESSION = 259
TAG_STRIP_OFFSETS = 273
TAG_SAMPLES_PER_PIXEL = 277
TAG_ROWS_PER_STRIP = 278
TAG_STRIP_BYTE_COUNTS = 279
TAG_PLANAR_CONFIG = 284
TAG_SAMPLE_FORMAT = 339
TAG_TILE_WIDTH = 322
TAG_TILE_OFFSETS = 324

_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}
_TYPE_FMT = {1: 'B', 2: 'B', 3: 'H', 4: 'I', 6: 'b', 7: 'B', 8: 'h', 9: 'i', 11: 'f', 12: 'd'}


class RawTiffError(RuntimeError):
    pass


def _read_ifd(fh):
    fh.seek(0)
    header = fh.read(8)
    if header[:2] == b'II':
        bo = '<'
    elif header[:2] == b'MM':
        bo = '>'
    else:
        raise RawTiffError('not a TIFF file')
    version = struct.unpack(bo + 'H', header[2:4])[0]
    if version != 42:
        raise RawTiffError(f'unsupported TIFF version {version} (BigTIFF is not handled)')

    ifd_offset = struct.unpack(bo + 'I', header[4:8])[0]
    fh.seek(ifd_offset)
    num_entries = struct.unpack(bo + 'H', fh.read(2))[0]
    entries = fh.read(12 * num_entries)

    tags = {}
    for i in range(num_entries):
        entry = entries[12 * i:12 * i + 12]
        tag, dtype, count = struct.unpack(bo + 'HHI', entry[:8])
        if dtype not in _TYPE_FMT:
            continue
        size = _TYPE_SIZE[dtype] * count
        if size <= 4:
            raw = entry[8:8 + size]
        else:
            pointer = struct.unpack(bo + 'I', entry[8:12])[0]
            fh.seek(pointer)
            raw = fh.read(size)
        tags[tag] = struct.unpack(bo + _TYPE_FMT[dtype] * count, raw)
    return bo, tags


class RawTiff:
    """按行随机访问一个无压缩 TIFF。"""

    def __init__(self, path):
        self.path = str(path)
        with open(self.path, 'rb') as fh:
            self.byteorder, tags = _read_ifd(fh)

        if TAG_TILE_WIDTH in tags or TAG_TILE_OFFSETS in tags:
            raise RawTiffError(f'{self.path}: tiled TIFF is not supported')
        compression = tags.get(TAG_COMPRESSION, (1,))[0]
        if compression != 1:
            raise RawTiffError(f'{self.path}: compression={compression} is not supported')
        planar = tags.get(TAG_PLANAR_CONFIG, (1,))[0]
        if planar != 1:
            raise RawTiffError(f'{self.path}: PlanarConfig={planar} is not supported')

        self.width = tags[TAG_IMAGE_WIDTH][0]
        self.height = tags[TAG_IMAGE_LENGTH][0]
        self.bands = tags.get(TAG_SAMPLES_PER_PIXEL, (1,))[0]

        bits = set(tags[TAG_BITS_PER_SAMPLE])
        fmts = set(tags.get(TAG_SAMPLE_FORMAT, (1,) * self.bands))
        if len(bits) != 1 or len(fmts) != 1:
            raise RawTiffError(f'{self.path}: mixed sample types are not supported')
        bit, fmt = bits.pop(), fmts.pop()
        kind = {1: 'u', 2: 'i', 3: 'f'}.get(fmt)
        if kind is None or bit % 8:
            raise RawTiffError(f'{self.path}: unsupported sample format {fmt}/{bit}')
        self.dtype = np.dtype(f'{self.byteorder}{kind}{bit // 8}')

        self.compression = compression
        self.rows_per_strip = min(tags.get(TAG_ROWS_PER_STRIP, (self.height,))[0], self.height)
        self.strip_offsets = np.asarray(tags[TAG_STRIP_OFFSETS], dtype=np.int64)
        self.strip_byte_counts = np.asarray(tags[TAG_STRIP_BYTE_COUNTS], dtype=np.int64)

        expected = self.rows_per_strip * self.width * self.bands * self.dtype.itemsize
        last = (self.height - (len(self.strip_offsets) - 1) * self.rows_per_strip)
        last_expected = last * self.width * self.bands * self.dtype.itemsize
        ok = np.all(self.strip_byte_counts[:-1] == expected) and self.strip_byte_counts[-1] == last_expected
        if not ok:
            raise RawTiffError(f'{self.path}: unexpected StripByteCounts layout')

        deltas = np.diff(self.strip_offsets)
        self.is_contiguous = bool(np.all(deltas == self.strip_byte_counts[:-1]))
        self.data_offset = int(self.strip_offsets[0])

    @property
    def shape(self):
        return (self.height, self.width, self.bands)

    def memmap(self):
        """仅在 strip 升序连续时可用，否则返回 None。"""
        if not self.is_contiguous:
            return None
        return np.memmap(self.path, dtype=self.dtype, mode='r',
                         offset=self.data_offset, shape=self.shape)

    def read_strided(self, row_step=1, col_step=1):
        """按步长抽样读取，只触碰被抽到的行，用于统计而不是训练。"""
        rows = np.arange(0, self.height, row_step)
        if self.is_contiguous:
            mm = np.memmap(self.path, dtype=self.dtype, mode='r',
                           offset=self.data_offset, shape=self.shape)
            return np.array(mm[::row_step, ::col_step])

        out = np.empty((len(rows), (self.width + col_step - 1) // col_step, self.bands),
                       dtype=self.dtype)
        rps = self.rows_per_strip
        row_bytes = self.width * self.bands * self.dtype.itemsize
        with open(self.path, 'rb') as fh:
            for i, r in enumerate(rows):
                s = r // rps
                fh.seek(int(self.strip_offsets[s]) + (r % rps) * row_bytes)
                buf = np.frombuffer(fh.read(row_bytes), dtype=self.dtype)
                out[i] = buf.reshape(self.width, self.bands)[::col_step]
        return out

    def read(self, row_start=0, row_stop=None, col_start=0, col_stop=None):
        """读取 [row_start, row_stop) x [col_start, col_stop)，返回 (rows, cols, C)。

        影像是 pixel-interleaved 的，一行里连续列的字节也是连续的，所以裁剪时
        限定列范围能少读将近一半——256 宽的裁剪块只需要 29MB 而不是整行的 59MB。
        """
        row_stop = self.height if row_stop is None else row_stop
        col_stop = self.width if col_stop is None else col_stop
        if not (0 <= row_start < row_stop <= self.height):
            raise ValueError(f'invalid row range [{row_start}, {row_stop})')
        if not (0 <= col_start < col_stop <= self.width):
            raise ValueError(f'invalid col range [{col_start}, {col_stop})')

        if self.is_contiguous:
            mm = np.memmap(self.path, dtype=self.dtype, mode='r',
                           offset=self.data_offset, shape=self.shape)
            return np.array(mm[row_start:row_stop, col_start:col_stop])

        n_col = col_stop - col_start
        out = np.empty((row_stop - row_start, n_col, self.bands), dtype=self.dtype)
        rps = self.rows_per_strip
        item = self.dtype.itemsize
        row_bytes = self.width * self.bands * item
        px_bytes = self.bands * item
        with open(self.path, 'rb') as fh:
            for i, r in enumerate(range(row_start, row_stop)):
                s = r // rps
                fh.seek(int(self.strip_offsets[s]) + (r % rps) * row_bytes + col_start * px_bytes)
                buf = np.frombuffer(fh.read(n_col * px_bytes), dtype=self.dtype)
                out[i] = buf.reshape(n_col, self.bands)
        return out


def imread(path, squeeze_single_band=True):
    """读取整幅影像，(H, W, C)；单波段且 squeeze 时返回 (H, W)。"""
    tif = RawTiff(path)
    arr = tif.read()
    if squeeze_single_band and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return arr


# 提交文件必须是无压缩 TIFF。CodaLab 评测端的 tifffile 没有 imagecodecs，
# LZW（压缩码 5）会直接读失败，日志表现为「没有匹配到影像」。
_SAMPLE_FORMAT = {'u': 1, 'i': 2, 'f': 3}
_GDAL_NODATA = 42113


def _ifd_payload(dtype, count, value):
    """把 tag 的值填进 IFD 条目末尾 4 字节。count*size 必须 <= 4。"""
    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
        if len(raw) > 4:
            raise ValueError('IFD inline value longer than 4 bytes')
        return raw.ljust(4, b'\x00')
    if dtype == 3:
        return struct.pack('<HH', int(value), 0)
    if dtype == 4:
        return struct.pack('<I', int(value))
    raise ValueError(f'unsupported TIFF dtype {dtype}')


def imwrite(path, array):
    """写一幅单波段无压缩 TIFF，(H, W) 二维数组。

    布局跟官方 Train_Labels 一致：IFD 紧挨 8 字节文件头，像素数据在 IFD 之后，
    Compression=1（无压缩）。不要改成 LZW——评测端 tifffile 没有 imagecodecs。
    """
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f'imwrite 只支持二维数组，收到 {array.shape}')
    kind = _SAMPLE_FORMAT.get(array.dtype.kind)
    if kind is None or array.dtype.itemsize not in (1, 2, 4, 8):
        raise ValueError(f'unsupported dtype {array.dtype}')
    if array.shape[0] > 65535:
        raise ValueError(f'height {array.shape[0]} 超出 SHORT RowsPerStrip 范围')

    array = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder('<'))
    height, width = array.shape
    data = array.tobytes()

    n_tags = 12
    ifd_offset = 8
    ifd_size = 2 + 12 * n_tags + 4
    data_offset = ifd_offset + ifd_size

    tags = [
        (TAG_IMAGE_WIDTH, 3, 1, width),
        (TAG_IMAGE_LENGTH, 3, 1, height),
        (TAG_BITS_PER_SAMPLE, 3, 1, array.dtype.itemsize * 8),
        (TAG_COMPRESSION, 3, 1, 1),          # 1 = 无压缩，禁止改成 5 (LZW)
        (262, 3, 1, 1),                      # PhotometricInterpretation = BlackIsZero
        (TAG_STRIP_OFFSETS, 4, 1, data_offset),
        (TAG_SAMPLES_PER_PIXEL, 3, 1, 1),
        (TAG_ROWS_PER_STRIP, 3, 1, height),  # 官方标签用 SHORT
        (TAG_STRIP_BYTE_COUNTS, 4, 1, len(data)),
        (TAG_PLANAR_CONFIG, 3, 1, 1),
        (TAG_SAMPLE_FORMAT, 3, 1, kind),
        (_GDAL_NODATA, 2, 2, b'0\x00'),      # 与官方标签一致，nodata = "0"
    ]

    body = struct.pack('<H', n_tags)
    for tag, dtype, count, value in sorted(tags):
        body += struct.pack('<HHI', tag, dtype, count)
        body += _ifd_payload(dtype, count, value)
    body += struct.pack('<I', 0)
    if len(body) != ifd_size:
        raise RuntimeError(f'IFD 大小 {len(body)}，预期 {ifd_size}')

    with open(path, 'wb') as fh:
        fh.write(struct.pack('<2sHI', b'II', 42, ifd_offset))
        fh.write(body)
        fh.write(data)
