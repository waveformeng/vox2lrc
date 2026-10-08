"""vox2lrc: isolated vocal stem in, timed lyrics (.lrc + word-level JSON) out."""

from .lines import Line, group_lines
from .lrc import to_json, to_lrc
from .pipeline import transcribe_stem
from .types import Word

__all__ = ["Line", "Word", "group_lines", "to_json", "to_lrc", "transcribe_stem"]
