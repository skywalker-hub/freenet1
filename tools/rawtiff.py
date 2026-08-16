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

    def read(self, row_start=0, row_stop=None):
        """读取 [row_start, row_stop) 行，返回 (rows, W, C)，行序始终正确。"""
        row_stop = self.height if row_stop is None else row_stop
        if not (0 <= row_start < row_stop <= self.height):
            raise ValueError(f'invalid row range [{row_start}, {row_stop})')

        if self.is_contiguous:
            mm = np.memmap(self.path, dtype=self.dtype, mode='r',
                           offset=self.data_offset, shape=self.shape)
            return np.array(mm[row_start:row_stop])

        out = np.empty((row_stop - row_start, self.width, self.bands), dtype=self.dtype)
        rps = self.rows_per_strip
        first, last = row_start // rps, (row_stop - 1) // rps
        with open(self.path, 'rb') as fh:
            for s in range(first, last + 1):
                fh.seek(int(self.strip_offsets[s]))
                buf = np.frombuffer(fh.read(int(self.strip_byte_counts[s])), dtype=self.dtype)
                rows_here = len(buf) // (self.width * self.bands)
                buf = buf.reshape(rows_here, self.width, self.bands)
                src_lo = max(row_start - s * rps, 0)
                src_hi = min(row_stop - s * rps, rows_here)
                dst_lo = s * rps + src_lo - row_start
                out[dst_lo:dst_lo + (src_hi - src_lo)] = buf[src_lo:src_hi]
        return out


def imread(path, squeeze_single_band=True):
    """读取整幅影像，(H, W, C)；单波段且 squeeze 时返回 (H, W)。"""
    tif = RawTiff(path)
    arr = tif.read()
    if squeeze_single_band and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return arr
