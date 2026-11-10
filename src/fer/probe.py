"""Inspect real media file properties — never assume container format,
frame rate, or sample rate; read them from the file itself.

Video is probed with PyAV (binds libavformat/libavcodec directly, so it
doesn't depend on a system ffmpeg install or on OpenCV's build flags for
.flv support). Audio is probed with soundfile (libsndfile).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import av
import soundfile as sf


@dataclass
class VideoProbe:
    readable: bool
    duration_s: float | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    n_frames: int | None = None
    video_codec: str | None = None
    has_audio_stream: bool | None = None
    error: str | None = None


@dataclass
class AudioProbe:
    readable: bool
    duration_s: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    subtype: str | None = None
    n_frames: int | None = None
    error: str | None = None


def probe_video(path: Path) -> VideoProbe:
    try:
        with av.open(str(path)) as container:
            vstreams = [s for s in container.streams if s.type == "video"]
            astreams = [s for s in container.streams if s.type == "audio"]
            if not vstreams:
                return VideoProbe(readable=False, error="no video stream in container")
            vs = vstreams[0]
            duration_s = None
            if container.duration is not None:
                duration_s = float(container.duration) / 1_000_000.0
            elif vs.duration is not None and vs.time_base is not None:
                duration_s = float(vs.duration * vs.time_base)

            fps = None
            if vs.average_rate is not None:
                fps = float(vs.average_rate)
            elif vs.guessed_rate is not None:
                fps = float(vs.guessed_rate)

            n_frames = vs.frames if vs.frames else None

            return VideoProbe(
                readable=True,
                duration_s=duration_s,
                fps=fps,
                width=vs.codec_context.width if vs.codec_context else None,
                height=vs.codec_context.height if vs.codec_context else None,
                n_frames=n_frames,
                video_codec=vs.codec_context.name if vs.codec_context else None,
                has_audio_stream=len(astreams) > 0,
            )
    except Exception as e:  # noqa: BLE001 - we must report, not raise, on bad media
        return VideoProbe(readable=False, error=f"{type(e).__name__}: {e}")


def probe_audio(path: Path) -> AudioProbe:
    try:
        info = sf.info(str(path))
        return AudioProbe(
            readable=True,
            duration_s=info.frames / info.samplerate if info.samplerate else None,
            sample_rate=info.samplerate,
            channels=info.channels,
            subtype=info.subtype,
            n_frames=info.frames,
        )
    except Exception as e:  # noqa: BLE001
        return AudioProbe(readable=False, error=f"{type(e).__name__}: {e}")
