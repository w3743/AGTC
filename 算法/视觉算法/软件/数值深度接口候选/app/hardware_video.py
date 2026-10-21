"""Native RKISP -> DMA-BUF -> MPP H.264 -> fragmented MP4 streaming.

Only compressed bytes enter Python. Each HTTP reader has its own cursor;
bounded history disconnects slow readers without blocking the camera.
"""
from collections import deque
import struct
import threading
import time


def fragment_info(box):
    """Read decode time and first-sample sync flag from one moof/traf."""
    def boxes(data):
        offset = 0
        while offset + 8 <= len(data):
            size, kind = struct.unpack_from('>I4s', data, offset)
            if size < 8 or offset + size > len(data):
                raise ValueError('Malformed MP4 fragment')
            yield kind, data[offset+8:offset+size]
            offset += size
    default_flags = None
    first_flags = None
    decode_time = None
    for kind, moof in boxes(box):
        if kind != b'moof':
            continue
        for kind, traf in boxes(moof):
            if kind != b'traf':
                continue
            for kind, payload in boxes(traf):
                flags = int.from_bytes(payload[1:4], 'big')
                if kind == b'tfhd':
                    offset = 8
                    for flag, size in ((1,8),(2,4),(8,4),(16,4)):
                        if flags & flag:
                            offset += size
                    if flags & 32:
                        default_flags = struct.unpack_from('>I', payload, offset)[0]
                elif kind == b'tfdt':
                    decode_time = struct.unpack_from('>Q' if payload[0] else '>I',payload,4)[0]
                elif kind == b'trun':
                    offset = 8 + (4 if flags & 1 else 0)
                    if flags & 4:
                        first_flags = struct.unpack_from('>I',payload,offset)[0]
                    elif flags & 1024:
                        offset += (4 if flags & 256 else 0) + (4 if flags & 512 else 0)
                        first_flags = struct.unpack_from('>I',payload,offset)[0]
    flags = first_flags if first_flags is not None else default_flags
    return decode_time, flags is not None and not bool(flags & 0x10000)


class FragmentBuffer:
    def __init__(self, max_fragments=8, max_bytes=16 * 1024 * 1024):
        self.condition = threading.Condition()
        self.pending = bytearray()
        self.header = b''
        self.init = None
        self.moof = None
        self.fragments = deque()
        self.max_fragments, self.max_bytes = max_fragments, max_bytes
        self.total_bytes = 0
        self.sequence = 0
        self.error = None
        self.closed = False
        self.first_decode_time = None

    def feed(self, data):
        with self.condition:
            self.pending.extend(data)
            while len(self.pending) >= 8:
                size, kind = struct.unpack_from('>I4s', self.pending)
                if size == 1:
                    if len(self.pending) < 16:
                        break
                    size = struct.unpack_from('>Q', self.pending, 8)[0]
                if size < 8 or size > self.max_bytes:
                    raise ValueError('Invalid or oversized MP4 box')
                if len(self.pending) < size:
                    break
                box = bytes(self.pending[:size])
                del self.pending[:size]
                if kind == b'ftyp':
                    self.header = box
                elif kind == b'moov':
                    self.init = self.header + box
                elif kind == b'moof':
                    self.moof = box
                    decode_time, self.sync = fragment_info(box)
                    if self.first_decode_time is None:
                        self.first_decode_time = decode_time
                elif kind == b'mdat':
                    if self.moof is None:
                        raise ValueError('MP4 media data without fragment header')
                    fragment = self.moof + box
                    self.moof = None
                    self.sequence += 1
                    self.fragments.append((self.sequence, fragment, self.sync))
                    self.total_bytes += len(fragment)
                    while (len(self.fragments) > self.max_fragments or
                           self.total_bytes > self.max_bytes):
                        self.total_bytes -= len(self.fragments.popleft()[1])
                self.condition.notify_all()

    def begin(self, timeout=4):
        with self.condition:
            self.condition.wait_for(lambda: (self.init and any(row[2] for row in self.fragments)) or self.closed,
                                    timeout=timeout)
            if self.closed or not self.init or not any(row[2] for row in self.fragments):
                raise RuntimeError(self.error or '4K encoder is starting')
            # Timestamp rounding or dropped input can create short P-only
            # fragments. New readers must begin at a verified sync sample.
            sequence, fragment, _ = next(row for row in reversed(self.fragments) if row[2])
            return self.init, sequence, fragment

    def after(self, sequence, timeout=2):
        with self.condition:
            self.condition.wait_for(lambda: self.sequence > sequence or self.closed,
                                    timeout=timeout)
            if self.closed:
                raise RuntimeError(self.error or '4K pipeline restarted')
            if self.fragments and sequence < self.fragments[0][0] - 1:
                raise RuntimeError('4K client fell behind; reconnect at the live edge')
            return next((row[:2] for row in self.fragments if row[0] > sequence), None)

    def close(self, error=None):
        with self.condition:
            self.closed, self.error = True, error
            self.condition.notify_all()


class HardwareVideo:
    WIDTH, HEIGHT = 3840, 2160

    def __init__(self, Gst, bitrate=10000000):
        self.Gst = Gst
        self.bitrate = max(2000000, min(30000000, int(bitrate)))
        self.streams = [FragmentBuffer(), FragmentBuffer()]
        self.frames = [deque(maxlen=300), deque(maxlen=300)]
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.threads = []
        self.pipeline = None
        self.error = None
        self.first_pts = [None, None]

    def branch(self, device, index, raw_rate=None):
        source = (
            f'v4l2src device={device} io-mode=dmabuf ! '
            'video/x-raw,format=NV12,width=3840,height=2160,framerate=30/1 ! ')
        # Keep the legacy encoder launch string byte-for-byte unchanged when
        # the optional low-rate raw ROI tee is not requested.
        encoder = (
            'queue max-size-buffers=4 max-size-bytes=0 max-size-time=0 leaky=downstream ! '
            f'mpph264enc bps={self.bitrate} gop=15 level=51 max-pending=4 max-reenc=0 '
            'header-mode=each-idr ! '
            f'h264parse name=encoded{index} ! video/x-h264,stream-format=avc,alignment=au ! '
            'mp4mux fragment-duration=500 streamable=true trak-timescale=90000 ! '
            f'appsink name=mp4_{index} max-buffers=32 drop=false sync=false')
        if raw_rate is None:
            return source + encoder
        if isinstance(raw_rate, bool) or not isinstance(raw_rate, int) or not 1 <= raw_rate <= 30:
            raise ValueError('raw_rate must be an integer between 1 and 30 fps')
        # raw_rate enables the raw tee and is validated here for symmetry with
        # NativeROIDepth.rate; native PTS bucketing performs the actual throttle.
        return (
            source + f'tee name=native{index} '
            f'native{index}. ! {encoder} '
            f'native{index}. ! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 '
            f'leaky=downstream ! appsink name=far{index} max-buffers=1 drop=true sync=false')

    def start(self, pipeline):
        self.pipeline = pipeline
        for index in range(2):
            pipeline.get_by_name(f'encoded{index}').get_static_pad('src').add_probe(
                self.Gst.PadProbeType.BUFFER, self._frame, index)
            thread = threading.Thread(target=self._consume, args=(index,),
                                      name=f'mpp-mp4-{index}', daemon=True)
            self.threads.append(thread)
            thread.start()

    def _frame(self, pad, info, index):
        buf = info.get_buffer()
        if buf is not None:
            clock = self.pipeline.get_clock()
            running = clock.get_time() - self.pipeline.get_base_time() if clock else None
            latency = (max(0, running - buf.pts) / 1e6
                       if running is not None and buf.pts != self.Gst.CLOCK_TIME_NONE else None)
            with self.lock:
                if self.first_pts[index] is None:
                    self.first_pts[index] = int(buf.pts)
                self.frames[index].append((time.monotonic(), int(buf.pts),
                                           int(buf.get_size()), latency))
        return self.Gst.PadProbeReturn.OK

    def _consume(self, index):
        sink = self.pipeline.get_by_name(f'mp4_{index}')
        try:
            while not self.stopped.is_set():
                sample = sink.emit('try-pull-sample', self.Gst.SECOND // 2)
                if sample is None:
                    continue
                buf = sample.get_buffer()
                self.streams[index].feed(buf.extract_dup(0, buf.get_size()))
        except Exception as exc:
            self.error = str(exc)
            self.streams[index].close(self.error)

    def status(self):
        output = []
        with self.lock:
            snapshots = [list(frames) for frames in self.frames]
        for index, frames in enumerate(snapshots):
            age = (time.monotonic() - frames[-1][0]) if frames else None
            duration = frames[-1][0] - frames[0][0] if len(frames) > 1 else 0
            output.append(dict(
                eye=index, resolution_wh=[self.WIDTH, self.HEIGHT],
                encoded_fps=round((len(frames) - 1) / duration, 3)
                if duration and age < 1 else 0,
                pts_fps=round((len(frames)-1)*1e9/(frames[-1][1]-frames[0][1]), 3)
                if len(frames)>1 and frames[-1][1]>frames[0][1] else None,
                age_ms=round(age * 1000) if age is not None else None,
                bitrate_mbps=round(sum(row[2] for row in frames[1:])*8/duration/1e6, 3)
                if duration else None,
                encode_latency_ms=round(frames[-1][3], 2)
                if frames and frames[-1][3] is not None else None,
                fragments=self.streams[index].sequence,
                buffered_bytes=self.streams[index].total_bytes))
            output[-1]['source_offset_s'] = (
                self.first_pts[index]/1e9 - self.streams[index].first_decode_time/90000
                if self.first_pts[index] is not None and self.streams[index].first_decode_time is not None else None)
        return dict(enabled=True, codec='H.264 High 5.1', transport='fragmented MP4 / MSE',
                    encoder='RK3588 MPP', native=True, streams=output, error=self.error)

    def source_time_ns(self, media_time, index=0):
        first_pts = self.first_pts[index]
        first_decode_time = self.streams[index].first_decode_time
        if first_pts is None or first_decode_time is None:
            raise ValueError('4K timestamp mapping is not ready')
        return int((float(media_time)-first_decode_time/90000)*1e9) + first_pts

    def close(self):
        self.stopped.set()
        for stream in self.streams:
            stream.close()
        for thread in self.threads:
            thread.join(1)
