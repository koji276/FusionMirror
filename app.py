# FUSIONDRIVER RC Architecture Note
# FusionMirror v0.1 is intentionally split into:
# - rebar: config.json for thresholds, device contracts, and persisted runtime state
# - concrete: app.py for audio capture, signal gating, Whisper inference, and subtitle UI
# This separation keeps policy and execution independent so later developers can tune behavior
# without rewriting the operational pipeline.
#
# RC tuning workflow for future maintainers:
# 1. Adjust `config.json -> audio.device` to bind the concrete logic to the
#    intended capture hardware without editing Python code.
# 2. Adjust `config.json -> audio.target_channel_index` if the counterpart audio
#    is routed to a different channel than the default Right channel.
# 3. Adjust `config.json -> vad.rms_threshold` first when speech is missed or
#    noise leaks through. This is the main gate-sensitivity control.
# 4. Adjust `config.json -> vad.min_speech_duration_ms` and
#    `config.json -> vad.max_silence_duration_ms` only after the RMS threshold
#    is behaving correctly, because these values shape segmentation latency and
#    pause tolerance rather than raw sensitivity.
# 5. Keep environment-specific workarounds, such as Streamlit output
#    suppression during model load, inside this Python layer so the JSON rebar
#    remains a stable contract rather than a collection of UI-specific hacks.

from __future__ import annotations

import contextlib
import io
import json
import os
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
import streamlit as st
import tqdm
import whisper


CONFIG_PATH = Path(__file__).with_name("config.json")


def update_config_section(section: str, updates: dict, config_path: Path = CONFIG_PATH) -> dict:
    """Merge updates into a named top-level section of config.json.

    Args:
        section: Top-level config section name such as `audio` or `state`.
        updates: Partial dictionary to merge into the target section.
        config_path: Destination JSON path.

    Returns:
        The refreshed configuration after applying the updates.
    """
    config = load_config(config_path)
    config.setdefault(section, {}).update(updates)
    save_config(config, config_path)
    return config


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load the RC rebar definition from config.json.

    Args:
        config_path: Absolute path to the JSON file that stores operational
            settings and persisted state.

    Returns:
        Parsed configuration dictionary.
    """
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_config(config: dict, config_path: Path = CONFIG_PATH) -> None:
    """Persist the full configuration, including runtime state, back to disk.

    Args:
        config: Configuration dictionary to be written.
        config_path: Destination JSON path.
    """
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)


def update_config_state(updates: dict, config_path: Path = CONFIG_PATH) -> dict:
    """Merge runtime state updates into config.json.

    Args:
        updates: Partial state dictionary such as the latest RMS or transcript.
        config_path: Destination JSON path.

    Returns:
        The refreshed configuration after applying the state updates.
    """
    config = load_config(config_path)
    config.setdefault("state", {}).update(updates)
    save_config(config, config_path)
    return config


def extract_channel(indata: np.ndarray, target_channel_index: int) -> np.ndarray:
    """Extract a specific channel from a multichannel input block.

    Args:
        indata: Audio block from the input stream with shape
            `(frame_count, channel_count)`.
        target_channel_index: Zero-based channel index to isolate.

    Returns:
        One-dimensional float32 array containing only the selected channel.

    Raises:
        ValueError: If the incoming block does not contain the requested
            channel.
    """
    if indata.ndim != 2 or indata.shape[1] <= target_channel_index:
        raise ValueError(
            f"Expected stereo-like input with channel index {target_channel_index}, "
            f"but received shape {indata.shape}."
        )
    return np.asarray(indata[:, target_channel_index], dtype=np.float32).copy()


def extract_right_channel(indata: np.ndarray, target_channel_index: int) -> np.ndarray:
    """Extract only the designated stereo channel from an input block.

    The business rule for FusionMirror is explicit: the counterpart's voice is
    assumed to be routed to the right channel of a stereo capture device.
    `sounddevice` provides each block as a 2-D NumPy array shaped
    `(frames, channels)`. Selecting `[:, 1]` therefore means:
    - keep every time-domain sample for the current block
    - discard the left-channel samples entirely
    - preserve the amplitude contour of the remote party only

    This function keeps the logic configurable through `target_channel_index`
    so the system can be retargeted if the hardware routing changes later.

    Args:
        indata: Audio block from the input stream with shape
            `(frame_count, channel_count)`.
        target_channel_index: Zero-based channel index to isolate. `1` means
            the right channel under the standard stereo convention.

    Returns:
        One-dimensional float32 array containing only the selected channel.

    Raises:
        ValueError: If the incoming block does not contain the requested
            channel.
    """
    return extract_channel(indata, target_channel_index)


def normalize_input_channels(indata: np.ndarray) -> np.ndarray:
    """Normalize incoming audio blocks so downstream logic always sees 2 channels.

    FusionMirror's production assumption is stereo input with physically
    separated left/right wiring. That assumption is correct for the target
    hardware, but it blocks testing on environments that expose only a single
    microphone channel. The development intent here is explicit: future
    maintainers must be able to validate the RMS meters, the Right-channel
    gate, and the Whisper path even on laptops or CI-like test rigs that do not
    have the real split-input microphone chain attached.

    For that reason, monaural input is adapted as follows:
    - if the device already provides 2 or more channels, preserve the block
      unchanged so true L/R separation is not damaged
    - if the device provides exactly 1 channel, duplicate that waveform into
      both columns, producing an internal `(frames, 2)` matrix

    The duplication means the debug meters for Left and Right will move
    together, and the Right-channel gate can still open, which is sufficient
    for local pipeline testing without changing the stereo behavior on real
    split input hardware. In other words, mono duplication is a deliberate
    testability feature, not an attempt to simulate true physical separation.

    Args:
        indata: Audio block from the input stream with shape
            `(frame_count, channel_count)`.

    Returns:
        A 2-D NumPy array with at least two channels available to downstream
        logic.

    Raises:
        ValueError: If the incoming block is not a 2-D array or contains zero
            channels.
    """
    if indata.ndim != 2 or indata.shape[1] == 0:
        raise ValueError(f"Expected input shape (frames, channels), but received {indata.shape}.")

    if indata.shape[1] == 1:
        return np.repeat(indata.astype(np.float32, copy=False), 2, axis=1)

    return indata


def list_input_devices() -> list[dict]:
    """Return input-capable audio devices discoverable by sounddevice.

    The debug UI uses this list to prove that the selected hardware is a real
    input source before any transcription work starts. Stereo-capable devices
    are especially important because FusionMirror depends on left/right channel
    separation at the physical wiring level.

    Returns:
        List of dictionaries containing device index, label, and channel count.
    """
    devices: list[dict] = []
    for index, device in enumerate(sd.query_devices()):
        max_input_channels = int(device["max_input_channels"])
        if max_input_channels <= 0:
            continue
        devices.append(
            {
                "index": index,
                "name": str(device["name"]),
                "max_input_channels": max_input_channels,
                "is_stereo_ready": max_input_channels >= 2,
                "label": f"{index}: {device['name']} (inputs={max_input_channels})",
            }
        )
    return devices


@contextlib.contextmanager
def suppress_model_load_progress() -> None:
    """Suppress tqdm and stderr output during local Whisper model loading.

    Streamlit launches Python in an environment where progress-bar writes to the
    host output streams are not always compatible with terminal-style cursor
    control. Whisper's model loader uses `tqdm`, which writes progress updates
    to `stderr` by default. Under some Streamlit executions, those writes can
    raise `[Errno 22] Invalid argument`, even though the model file itself is
    otherwise loadable.

    The fix is intentionally local to the concrete layer:
    - disable `tqdm` while the model is loading
    - redirect `sys.stderr` to an in-memory buffer during that same window

    This keeps the RC contract clean. The JSON rebar continues to describe what
    model to load, while the Python logic absorbs how to load it safely in a
    Streamlit-hosted process.

    During the guarded section:
    - `tqdm.tqdm` is wrapped with `disable=True`
    - `sys.stderr` is redirected to an in-memory buffer

    This keeps `whisper.load_model(...)` deterministic even when the hosting
    terminal cannot accept progress-bar control output.
    """
    original_tqdm = tqdm.tqdm
    original_stderr = sys.stderr
    muted_stderr = io.StringIO()

    try:
        tqdm.tqdm = partial(original_tqdm, disable=True)
        sys.stderr = muted_stderr
        yield
    finally:
        tqdm.tqdm = original_tqdm
        sys.stderr = original_stderr


def calculate_rms(audio_chunk: np.ndarray) -> float:
    """Calculate RMS amplitude for a monaural audio block.

    RMS is used here as a simple physical proxy for acoustic energy. In
    practical terms:
    - a larger RMS usually means meaningful speech energy is present
    - a tiny RMS usually means room noise, circuit hiss, or silence

    The metric is appropriate for an MVP gate because it is cheap to compute
    per block and stable enough to suppress clearly non-speech low-level input
    before Whisper is invoked.

    Args:
        audio_chunk: One-dimensional waveform for a single channel.

    Returns:
        RMS amplitude as a Python float.
    """
    if audio_chunk.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio_chunk), dtype=np.float32)))


def is_above_gate(rms_value: float, threshold: float) -> bool:
    """Decide whether a block contains enough energy to be treated as speech.

    Args:
        rms_value: Measured RMS amplitude for the current block.
        threshold: Minimum RMS amplitude required to pass the gate.

    Returns:
        True when the block is considered speech-like enough to keep.
    """
    return rms_value >= threshold


def transcribe_with_whisper(
    model: whisper.Whisper,
    audio_array: np.ndarray,
    whisper_config: dict,
    dictionary_words: list[str],
) -> str:
    """Transcribe a buffered speech segment locally with Whisper.

    The audio entering this function has already passed the right-channel
    extraction and RMS gate, so the model only sees segments likely to contain
    the counterpart's speech. This keeps the local pipeline efficient and
    reduces subtitle churn caused by noise-only blocks.

    Args:
        model: Preloaded Whisper model instance.
        audio_array: Concatenated speech waveform sampled at the configured
            rate expected by Whisper.
        whisper_config: Whisper-related config section containing language,
            task, and optional base initial prompt.
        dictionary_words: Optional dynamic dictionary terms to bias decoding.

    Returns:
        Stripped transcription text. Empty string if no useful text was found.
    """
    if audio_array.size == 0:
        return ""

    base_prompt = str(whisper_config.get("initial_prompt", "")).strip()
    dictionary_prompt = ", ".join(str(word).strip() for word in dictionary_words if str(word).strip())
    initial_prompt = " ".join(part for part in (base_prompt, dictionary_prompt) if part)

    result = model.transcribe(
        audio_array.astype(np.float32),
        language=whisper_config["language"],
        task=whisper_config["task"],
        fp16=False,
        verbose=False,
        initial_prompt=initial_prompt,
    )
    return result.get("text", "").strip()


def restore_meaning(raw_text: str, config: dict) -> str:
    """Repair Whisper output with an LLM using the configured dictionary.

    If no LLM settings or API key are available, this function returns the
    original text unchanged so the transcription pipeline remains usable.

    Args:
        raw_text: Whisper transcription before semantic repair.
        config: Full runtime config including optional `dictionary` and `llm`.

    Returns:
        Corrected subtitle text when the LLM call succeeds, otherwise the
        original transcription.
    """
    raw_text = raw_text.strip()
    if not raw_text:
        return ""

    dictionary_items = [
        str(word).strip() for word in config.get("dictionary", []) if str(word).strip()
    ]
    dictionary_text = ", ".join(dictionary_items)
    prompt = (
        "あなたはビジネス会議の議事録補正アシスタントです。"
        "以下の【辞書】にある固有名詞やキーワードを参照し、"
        "【未修正テキスト】に含まれる音声認識の誤り（空耳）を正しく修復してください。"
        "意味を変えず、自然なビジネス英語に整えてください。\n"
        f"【辞書】: {dictionary_text}\n"
        f"【未修正テキスト】: {raw_text}"
    )

    llm_config = config.get("llm", {})
    api_key = str(llm_config.get("api_key") or os.environ.get("GEMINI_API_KEY", "")).strip()
    if not api_key:
        return raw_text

    model_name = str(llm_config.get("model", "gemini-1.5-flash")).strip() or "gemini-1.5-flash"
    timeout_seconds = float(llm_config.get("timeout_seconds", 10))
    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        f"?key={api_key}"
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt,
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": llm_config.get("temperature", 0.1),
        },
    }

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_data = json.load(response)
    except urllib.error.URLError:
        return raw_text
    except TimeoutError:
        return raw_text
    except Exception:
        return raw_text

    candidates = response_data.get("candidates", [])
    for candidate in candidates:
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = str(part.get("text", "")).strip()
            if text:
                return text
    return raw_text


@dataclass
class AudioSegmentCollector:
    """Accumulate right-channel speech until silence indicates segment end.

    The collector implements an RMS gate that behaves like a lightweight VAD:
    - blocks above the threshold are appended to the active speech segment
    - short silent gaps are tolerated so normal speech pauses do not split
      every clause
    - once silence exceeds the configured limit, the segment is closed and sent
      to transcription if it was long enough to be meaningful

    Attributes:
        min_speech_blocks: Minimum count of voiced blocks required before a
            segment can be emitted.
        max_silence_blocks: Maximum tolerated count of silent blocks after
            speech has started.
        speech_blocks: Buffered voiced blocks for the current segment.
        silence_blocks: Count of consecutive silent blocks after speech start.
    """

    min_speech_blocks: int
    max_silence_blocks: int
    speech_blocks: list[np.ndarray] = field(default_factory=list)
    silence_blocks: int = 0

    def process_block(self, chunk: np.ndarray, gate_open: bool) -> Optional[np.ndarray]:
        """Update the current segment with a new gated block.

        Args:
            chunk: Right-channel waveform for the current block.
            gate_open: True when the block RMS is above threshold.

        Returns:
            A completed speech segment when one closes, otherwise None.
        """
        if gate_open:
            self.speech_blocks.append(chunk)
            self.silence_blocks = 0
            return None

        if not self.speech_blocks:
            return None

        self.silence_blocks += 1
        if self.silence_blocks <= self.max_silence_blocks:
            self.speech_blocks.append(chunk)
            return None

        return self._flush()

    def flush(self) -> Optional[np.ndarray]:
        """Force-close the current segment.

        Returns:
            Completed speech segment if enough voiced content was accumulated.
        """
        return self._flush()

    def _flush(self) -> Optional[np.ndarray]:
        if len(self.speech_blocks) < self.min_speech_blocks:
            self.speech_blocks.clear()
            self.silence_blocks = 0
            return None

        segment = np.concatenate(self.speech_blocks).astype(np.float32, copy=False)
        self.speech_blocks.clear()
        self.silence_blocks = 0
        return segment


@dataclass
class AudioCallbackEvent:
    """Represent a lightweight event emitted from the realtime audio callback."""

    kind: str
    audio_block: Optional[np.ndarray] = None
    error_message: str = ""


class FusionMirrorEngine:
    """Coordinate audio capture, gating, local Whisper, and UI-facing state."""

    def __init__(self, config_path: Path = CONFIG_PATH) -> None:
        """Initialize the live engine from the RC configuration.

        Args:
            config_path: Path to the `config.json` rebar file.
        """
        self.config_path = config_path
        self.config = load_config(config_path)

        audio_cfg = self.config["audio"]
        vad_cfg = self.config["vad"]

        self.samplerate = int(audio_cfg["samplerate"])
        self.block_duration_ms = int(audio_cfg["block_duration_ms"])
        self.channels = int(audio_cfg["channels"])
        self.target_channel_index = int(audio_cfg["target_channel_index"])
        self.device = audio_cfg["device"]
        self.dtype = audio_cfg["dtype"]
        self.rms_threshold = float(vad_cfg["rms_threshold"])
        self.min_speech_duration_ms = int(vad_cfg["min_speech_duration_ms"])
        self.max_silence_duration_ms = int(vad_cfg["max_silence_duration_ms"])
        self.caption_limit = int(self.config["ui"]["caption_limit"])

        self.blocksize = max(1, int(self.samplerate * self.block_duration_ms / 1000))
        min_speech_blocks = max(1, int(np.ceil(self.min_speech_duration_ms / self.block_duration_ms)))
        max_silence_blocks = max(1, int(np.ceil(self.max_silence_duration_ms / self.block_duration_ms)))

        self.collector = AudioSegmentCollector(
            min_speech_blocks=min_speech_blocks,
            max_silence_blocks=max_silence_blocks,
        )
        self.audio_event_queue: queue.Queue[AudioCallbackEvent] = queue.Queue()
        self.segment_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.state_queue: queue.Queue[Optional[dict[str, object]]] = queue.Queue()
        self.subtitle_queue: queue.Queue[str] = queue.Queue()

        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.audio_thread: Optional[threading.Thread] = None
        self.audio_processing_thread: Optional[threading.Thread] = None
        self.state_writer_thread: Optional[threading.Thread] = None
        self.transcription_thread: Optional[threading.Thread] = None
        self.model: Optional[whisper.Whisper] = None
        self.stream: Optional[sd.InputStream] = None

        self.latest_rms = float(self.config["state"].get("last_rms", 0.0))
        self.latest_left_rms = float(self.config["state"].get("last_left_rms", 0.0))
        self.latest_right_rms = float(self.config["state"].get("last_right_rms", self.latest_rms))
        self.latest_transcript = self.config["state"].get("last_transcript", "")
        self.latest_error = self.config["state"].get("last_error", "")
        self.captions: list[str] = []

    def start(self) -> None:
        """Start audio capture and transcription threads if not already active.

        The startup path deliberately loads Whisper before audio capture
        threads begin. That ordering prevents the UI from entering a "capture is
        live but transcription is not ready" state. Model loading is wrapped by
        `suppress_model_load_progress()` so Streamlit-specific `tqdm` output
        failures are contained here instead of leaking into the user-visible
        control flow.
        """
        if self.is_running:
            return

        self.stop_event.clear()
        with suppress_model_load_progress():
            self.model = whisper.load_model(self.config["whisper"]["model_name"])
        self.state_writer_thread = threading.Thread(target=self._run_state_writer_loop, daemon=True)
        self.audio_processing_thread = threading.Thread(target=self._run_audio_processing_loop, daemon=True)
        self.audio_thread = threading.Thread(target=self._run_audio_stream, daemon=True)
        self.transcription_thread = threading.Thread(target=self._run_transcription_loop, daemon=True)

        self.state_writer_thread.start()
        self.audio_processing_thread.start()
        self.audio_thread.start()
        self.transcription_thread.start()
        self._persist_state(session_active=True, last_error="")

    def stop(self) -> None:
        """Stop live capture and flush any remaining buffered audio."""
        has_active_workers = any(
            thread is not None and thread.is_alive()
            for thread in (
                self.audio_thread,
                self.audio_processing_thread,
                self.transcription_thread,
                self.state_writer_thread,
            )
        )
        if not has_active_workers:
            self._persist_state(session_active=False)
            return

        self.stop_event.set()

        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=3)

        if self.audio_processing_thread and self.audio_processing_thread.is_alive():
            self.audio_event_queue.put(AudioCallbackEvent(kind="stop"))

        if self.audio_processing_thread and self.audio_processing_thread.is_alive():
            self.audio_processing_thread.join(timeout=10)
        else:
            pending_segment = self.collector.flush()
            if pending_segment is not None and pending_segment.size > 0:
                self.segment_queue.put(pending_segment)
            self.segment_queue.put(np.array([], dtype=np.float32))

        if self.transcription_thread and self.transcription_thread.is_alive():
            self.transcription_thread.join(timeout=10)

        self._persist_state(session_active=False)

        self.state_queue.put(None)
        if self.state_writer_thread and self.state_writer_thread.is_alive():
            self.state_writer_thread.join(timeout=5)

        self.audio_processing_thread = None
        self.audio_thread = None
        self.state_writer_thread = None
        self.transcription_thread = None
        self.stream = None
        self.model = None

    @property
    def is_running(self) -> bool:
        """Return True while the engine has active worker threads."""
        return (
            self.audio_processing_thread is not None
            and self.audio_processing_thread.is_alive()
            and
            self.audio_thread is not None
            and self.audio_thread.is_alive()
            and self.transcription_thread is not None
            and self.transcription_thread.is_alive()
        )

    def drain_subtitles(self) -> list[str]:
        """Retrieve all newly transcribed subtitle lines for the UI.

        Returns:
            List of new caption strings produced since the last poll.
        """
        items: list[str] = []
        while True:
            try:
                items.append(self.subtitle_queue.get_nowait())
            except queue.Empty:
                break

        if items:
            with self.lock:
                self.captions.extend(items)
                self.captions = self.captions[-self.caption_limit :]
        return items

    def snapshot(self) -> dict:
        """Return a UI-safe snapshot of current runtime values.

        Returns:
            Dictionary containing run status, RMS, last transcript, recent
            captions, and latest error message.
        """
        self.drain_subtitles()
        with self.lock:
            return {
                "is_running": self.is_running,
                "latest_rms": self.latest_rms,
                "latest_left_rms": self.latest_left_rms,
                "latest_right_rms": self.latest_right_rms,
                "device": self.device,
                "latest_transcript": self.latest_transcript,
                "captions": list(self.captions),
                "latest_error": self.latest_error,
            }

    def _persist_state(self, **updates: object) -> None:
        """Queue selected runtime values for asynchronous config.json persistence."""
        with self.lock:
            state = self.config.setdefault("state", {})
            state.update(updates)
            if "last_rms" in updates:
                self.latest_rms = float(updates["last_rms"])
            if "last_left_rms" in updates:
                self.latest_left_rms = float(updates["last_left_rms"])
            if "last_right_rms" in updates:
                self.latest_right_rms = float(updates["last_right_rms"])
            if "last_transcript" in updates:
                self.latest_transcript = str(updates["last_transcript"])
            if "last_error" in updates:
                self.latest_error = str(updates["last_error"])

        state_writer_is_available = (
            self.state_writer_thread is not None
            and self.state_writer_thread.is_alive()
            and threading.current_thread() is not self.state_writer_thread
        )
        if state_writer_is_available:
            self.state_queue.put(dict(updates))
            return

        config = update_config_state(dict(updates), self.config_path)
        with self.lock:
            self.config = config
            self.latest_error = config["state"].get("last_error", self.latest_error)

    def set_device(self, device_index: Optional[int]) -> None:
        """Persist and apply a newly selected input device.

        The selected device index is written immediately into the RC rebar
        layer (`config.json`) so the hardware contract stays synchronized with
        the UI. If capture is already active, the engine is restarted so the
        new device takes effect without waiting for a manual relaunch.

        Args:
            device_index: `sounddevice` input device index or None to use the
                system default.
        """
        was_running = self.is_running
        if was_running:
            self.stop()

        config = update_config_section("audio", {"device": device_index}, self.config_path)
        with self.lock:
            self.config = config
            self.device = config["audio"]["device"]

        if was_running:
            self.start()

    def _handle_audio_block(self, indata: np.ndarray) -> None:
        """Process a stereo input block into gated right-channel segments.

        If the current device is monaural, the single waveform is duplicated
        into synthetic left/right channels before any RMS measurement is
        performed. This keeps the debugging and Whisper gate path testable on
        1-channel microphones while leaving genuine stereo separation intact.

        Operationally, this method is where the RC split becomes visible:
        - `config.json -> audio.target_channel_index` decides which channel is
          treated as the counterpart voice
        - `config.json -> vad.rms_threshold` decides whether the Right channel
          has enough acoustic energy to pass the gate
        - `config.json -> vad.*duration_ms` values decide how blocks are
          accumulated into Whisper-ready segments

        Future maintainers should tune those JSON values first before changing
        this logic. The intent is for hardware assumptions and gate sensitivity
        to remain policy, while this method remains execution.

        Args:
            indata: Raw block from the sounddevice callback.
        """
        normalized_indata = normalize_input_channels(indata)
        left_channel = extract_channel(normalized_indata, 0)
        right_channel = extract_right_channel(normalized_indata, self.target_channel_index)
        left_rms = calculate_rms(left_channel)
        right_rms = calculate_rms(right_channel)
        gate_open = is_above_gate(right_rms, self.rms_threshold)
        segment = self.collector.process_block(right_channel, gate_open)

        self._persist_state(
            last_rms=round(right_rms, 6),
            last_left_rms=round(left_rms, 6),
            last_right_rms=round(right_rms, 6),
        )

        if segment is not None and segment.size > 0:
            self.segment_queue.put(segment)

    def _audio_callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        """Receive live audio blocks from sounddevice.

        Args:
            indata: Incoming audio frame matrix from `sounddevice`.
            frames: Number of frames in the block.
            time_info: PortAudio timing data.
            status: Status flags raised by PortAudio.
        """
        del frames, time_info

        if status:
            self.audio_event_queue.put_nowait(AudioCallbackEvent(kind="status", error_message=str(status)))

        if self.stop_event.is_set():
            raise sd.CallbackStop()

        try:
            self.audio_event_queue.put_nowait(
                AudioCallbackEvent(kind="audio", audio_block=np.asarray(indata, dtype=np.float32).copy())
            )
        except Exception as exc:  # pragma: no cover - defensive path for hardware callbacks
            self.audio_event_queue.put_nowait(AudioCallbackEvent(kind="status", error_message=str(exc)))
            self.stop_event.set()
            raise sd.CallbackStop() from exc

    def _run_audio_processing_loop(self) -> None:
        """Process callback-queued audio blocks outside the realtime callback."""
        try:
            while True:
                event = self.audio_event_queue.get()
                if event.kind == "stop":
                    break

                if event.kind == "status":
                    self._persist_state(last_error=event.error_message)
                    continue

                if event.audio_block is None:
                    continue

                self._handle_audio_block(event.audio_block)
        except Exception as exc:
            self._persist_state(last_error=str(exc))
            self.stop_event.set()
        finally:
            pending_segment = self.collector.flush()
            if pending_segment is not None and pending_segment.size > 0:
                self.segment_queue.put(pending_segment)
            self.segment_queue.put(np.array([], dtype=np.float32))

    def _run_audio_stream(self) -> None:
        """Open the stereo capture stream and feed the callback loop."""
        try:
            with sd.InputStream(
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                device=self.device,
                channels=self.channels,
                dtype=self.dtype,
                callback=self._audio_callback,
            ) as stream:
                self.stream = stream
                while not self.stop_event.is_set():
                    time.sleep(0.1)
        except Exception as exc:
            self._persist_state(last_error=str(exc))
            self.stop_event.set()

    def _run_state_writer_loop(self) -> None:
        """Serialize runtime state writes so config.json I/O stays off realtime paths."""
        while True:
            updates = self.state_queue.get()
            if updates is None:
                break

            batched_updates = dict(updates)
            stop_requested = False
            while True:
                try:
                    next_updates = self.state_queue.get_nowait()
                except queue.Empty:
                    break

                if next_updates is None:
                    stop_requested = True
                    break
                batched_updates.update(next_updates)

            config = update_config_state(batched_updates, self.config_path)
            with self.lock:
                self.config = config
                self.latest_error = config["state"].get("last_error", self.latest_error)

            if stop_requested:
                break

    def _run_transcription_loop(self) -> None:
        """Convert completed speech segments into subtitle text with Whisper."""
        try:
            while True:
                segment = self.segment_queue.get()
                if segment.size == 0 and self.stop_event.is_set():
                    break

                if self.model is None:
                    continue

                text = transcribe_with_whisper(
                    model=self.model,
                    audio_array=segment,
                    whisper_config=self.config.get("whisper", {}),
                    dictionary_words=self.config.get("dictionary", []),
                )
                if not text:
                    continue

                restored_text = restore_meaning(text, self.config)
                final_text = restored_text or text

                with self.lock:
                    self.latest_transcript = final_text

                self.subtitle_queue.put(final_text)
                self._persist_state(last_transcript=final_text, last_error="")
        except Exception as exc:
            self._persist_state(last_error=str(exc))
            self.stop_event.set()


def get_engine() -> FusionMirrorEngine:
    """Create or reuse the session-scoped FusionMirror engine.

    Returns:
        The active engine stored in `st.session_state`.
    """
    if "fusionmirror_engine" not in st.session_state:
        st.session_state.fusionmirror_engine = FusionMirrorEngine()
    return st.session_state.fusionmirror_engine


def sync_selected_device(engine: FusionMirrorEngine) -> None:
    """Persist the device chosen in the Streamlit selectbox into config.json.

    Args:
        engine: Session-scoped live engine.
    """
    selected_value = st.session_state.get("selected_input_device")
    normalized_value = None if selected_value == "default" else int(selected_value)
    if normalized_value == engine.device:
        return
    engine.set_device(normalized_value)


def render_device_selector(engine: FusionMirrorEngine) -> None:
    """Render the input device selector and keep RC config synchronized.

    Args:
        engine: Session-scoped live engine.
    """
    devices = list_input_devices()
    options = ["default"] + [device["index"] for device in devices]
    format_map = {"default": "System default input device"}
    format_map.update({device["index"]: device["label"] for device in devices})

    if "selected_input_device" not in st.session_state:
        st.session_state.selected_input_device = "default" if engine.device is None else engine.device

    current_value = st.session_state.selected_input_device
    if current_value not in options:
        current_value = "default"
        st.session_state.selected_input_device = current_value

    selected_option = st.selectbox(
        "Input device",
        options=options,
        index=options.index(current_value),
        format_func=lambda option: format_map[option],
        key="selected_input_device",
        on_change=sync_selected_device,
        args=(engine,),
    )

    selected_device = next((device for device in devices if device["index"] == selected_option), None)
    if selected_device is None and selected_option != "default":
        st.warning("Selected device is no longer available.")
    elif selected_device is not None and not selected_device["is_stereo_ready"]:
        st.warning("This device has fewer than 2 input channels. L/R separation cannot be validated on it.")


def render_controls(engine: FusionMirrorEngine) -> None:
    """Render start and stop controls for the live session.

    Args:
        engine: Session-scoped live engine.
    """
    render_device_selector(engine)
    col_start, col_stop = st.columns(2)
    with col_start:
        if st.button("Start Capture", use_container_width=True, disabled=engine.is_running):
            engine.start()
    with col_stop:
        if st.button("Stop Capture", use_container_width=True, disabled=not engine.is_running):
            engine.stop()


def render_lr_debug_panel(snapshot: dict) -> None:
    """Render left/right RMS meters for physical channel separation testing.

    Args:
        snapshot: Runtime state snapshot from the engine.
    """
    st.subheader("L/R Separation Debug")
    st.caption("Use this mode to verify that your voice appears on Left and the counterpart appears on Right before Whisper runs.")

    left_rms = float(snapshot["latest_left_rms"])
    right_rms = float(snapshot["latest_right_rms"])
    rms_scale = 0.1

    col_left, col_right = st.columns(2)
    with col_left:
        st.write(f"Left RMS (self): `{left_rms:.6f}`")
        st.progress(min(left_rms / rms_scale, 1.0), text="Left channel energy")
    with col_right:
        st.write(f"Right RMS (counterpart): `{right_rms:.6f}`")
        st.progress(min(right_rms / rms_scale, 1.0), text="Right channel energy")


def render_live_body(engine: FusionMirrorEngine) -> None:
    """Render current engine status, RMS, and rolling subtitle history.

    Args:
        engine: Session-scoped live engine.
    """
    snapshot = engine.snapshot()
    status_text = "Running" if snapshot["is_running"] else "Stopped"
    debug_mode = st.checkbox(
        "L/R separation debug mode",
        value=st.session_state.get("lr_debug_mode", True),
        key="lr_debug_mode",
    )

    st.subheader("Live Status")
    st.write(f"Session: `{status_text}`")
    st.write(f"Input device index: `{snapshot['device']}`")
    st.write(f"Right-channel RMS for gate: `{snapshot['latest_rms']:.6f}`")

    if snapshot["latest_error"]:
        st.error(snapshot["latest_error"])

    if debug_mode:
        render_lr_debug_panel(snapshot)

    st.subheader("Latest English Subtitle")
    st.markdown(
        f"> {snapshot['latest_transcript']}" if snapshot["latest_transcript"] else "> Waiting for speech..."
    )

    st.subheader("Recent Subtitle History")
    if snapshot["captions"]:
        for caption in reversed(snapshot["captions"]):
            st.write(f"- {caption}")
    else:
        st.write("No subtitles yet.")


def build_live_panel(engine: FusionMirrorEngine, refresh_ms: int) -> Callable[[], None]:
    """Create a live panel function with automatic refresh when supported.

    Args:
        engine: Session-scoped live engine.
        refresh_ms: Desired refresh cadence in milliseconds.

    Returns:
        Callable that renders the live panel.
    """

    def _body() -> None:
        render_live_body(engine)

    if hasattr(st, "fragment"):
        return st.fragment(run_every=f"{refresh_ms}ms")(_body)
    return _body


def main() -> None:
    """Launch the Streamlit UI for local FusionMirror transcription."""
    config = load_config()
    st.set_page_config(page_title=config["ui"]["title"], layout="wide")
    st.title(config["ui"]["title"])
    st.caption("Local-only MVP for right-channel meeting subtitle capture with Whisper.")

    engine = get_engine()
    render_controls(engine)
    live_panel = build_live_panel(engine, int(config["ui"]["refresh_ms"]))
    live_panel()


if __name__ == "__main__":
    main()
