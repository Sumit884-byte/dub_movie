"""Character personality, tone detection, and context-aware translation."""

from __future__ import annotations

import re
import unicodedata

from fast_translate import translate_many, translate_text

_CAST_FIELDS = (
    "voice_type",
    "vocal_archetype",
    "key_characteristics",
    "role",
    "speaking_style",
    "translate_as",
    "tts_note",
)

# Voice + personality + dubbing translation style per cast member.
CHARACTER_PROFILES: dict[str, dict] = {
    "クレア": {
        "voice_type": "High-Soprano (Bright & Energetic)",
        "vocal_archetype": "The Spirited Princess / The Playful Whiner",
        "key_characteristics": (
            "Crisp, clear, youthful voice with high pitch flexibility; shifts into a "
            "slight melodious whine when bored without sounding grating. Energetic, "
            "expressive, and slightly fast-paced."
        ),
        "role": "10-year-old cheerful princess; playful, slightly demanding yet cute",
        "speaking_style": "bright teasing, playful whine, uses ~ and のじゃ; never cold commands",
        "translate_as": "playful girl coaxing a friend — warm, cute, not bossy instructions",
        "tts_note": "energetic and cute, light teasing",
        "voices": {
            "hi": {"voice": "hi-IN-SwaraNeural", "rate": "+18%", "pitch": "+30Hz"},
            "en": {"voice": "en-US-AriaNeural", "rate": "+14%", "pitch": "+28Hz"},
        },
    },
    "マイロ": {
        "voice_type": "Mid-Mezzo / Boy Soprano (Soft & Grounded)",
        "vocal_archetype": "The Focused Intellectual / The Quiet Creator",
        "key_characteristics": (
            "Calm, steady, gentle voice with a lower, grounded pitch than Claire. "
            "Slightly distracted, soft delivery to show he is preoccupied with art or thought."
        ),
        "role": "calm focused 10-year-old court painter",
        "speaking_style": "soft, gentle, distracted, preoccupied with art",
        "translate_as": "quiet gentle boy, often hesitant; use ... where Japanese has うん…",
        "tts_note": "soft and unhurried",
        "voices": {
            "hi": {"voice": "hi-IN-MadhurNeural", "rate": "+6%", "pitch": "+18Hz"},
            "en": {"voice": "en-US-ChristopherNeural", "rate": "+3%", "pitch": "+14Hz"},
        },
    },
    "のび太": {
        "voice_type": "High-Alto / Boy Soprano (Nasal & Vulnerable)",
        "vocal_archetype": "The Whiny Underdog",
        "key_characteristics": (
            "Highly expressive with a natural slight nasality. Wide emotional range from "
            "lazy low-energy dragging to high-pitched panicked squeaks when fleeing trouble."
        ),
        "role": "clumsy kind-hearted boy",
        "speaking_style": "whiny, vulnerable, easily panicked",
        "translate_as": "worried boy voice; pleading or flustered, not confident orders",
        "tts_note": "slightly whiny and anxious",
        "voices": {
            "hi": {"voice": "hi-IN-MadhurNeural", "rate": "+16%", "pitch": "+30Hz"},
            "en": {"voice": "en-US-DavisNeural", "rate": "+14%", "pitch": "+28Hz"},
        },
    },
    "しずか": {
        "voice_type": "Pure Soprano (Light & Breathed)",
        "vocal_archetype": "The Polite Girl-Next-Door",
        "key_characteristics": (
            "Soft, smooth, gentle with clean diction. Airy, warm tone with a natural smile. "
            "Highly polite and careful, avoiding sharp or harsh edges."
        ),
        "role": "polite responsible girl",
        "speaking_style": "sweet, gentle, well-mannered",
        "translate_as": "polite soft girl; respectful wording, no harshness",
        "tts_note": "warm and polite",
        "voices": {
            "hi": {"voice": "hi-IN-SwaraNeural", "rate": "+4%", "pitch": "+14Hz"},
            "en": {"voice": "en-US-JennyNeural", "rate": "+2%", "pitch": "+12Hz"},
        },
    },
    "ジャイアン": {
        "voice_type": "Deep Baritone (Gravelly & Boisterous)",
        "vocal_archetype": "The Rowdy Enforcer",
        "key_characteristics": (
            "Full-bodied, chest-resonant, rough around the edges. Vocal power for loud yelling, "
            "roaring laughter, and a booming, unrefined, imposing tone."
        ),
        "role": "tough loud bully",
        "speaking_style": "boisterous, deep, aggressive",
        "translate_as": "bold rough boy; strong verbs, not polite",
        "tts_note": "loud and forceful — tough boy, not adult baritone",
        "voices": {
            "hi": {"voice": "hi-IN-MadhurNeural", "rate": "-6%", "pitch": "-18Hz", "volume": "+10%"},
            "en": {"voice": "en-US-GuyNeural", "rate": "-8%", "pitch": "-20Hz", "volume": "+8%"},
        },
    },
    "スネ夫": {
        "voice_type": "Tenor (Sharp & Nasal)",
        "vocal_archetype": "The Braggart / The Slick Sycophant",
        "key_characteristics": (
            "Highly nasal, pinched, slippery pitch sitting higher in the throat for a smug or "
            "sneaky quality. Shifts from arrogant boastful drawl to squeaky flattery."
        ),
        "role": "wealthy boastful boy",
        "speaking_style": "sharp, nasal, sneaky, show-off",
        "translate_as": "smug boasting tone; slightly smirking delivery",
        "tts_note": "quick and boastful",
        "voices": {
            "hi": {"voice": "hi-IN-MadhurNeural", "rate": "+10%", "pitch": "+20Hz"},
            "en": {"voice": "en-US-BrianNeural", "rate": "+8%", "pitch": "+18Hz"},
        },
    },
    "ドラえもん": {
        "voice": "hi-IN-MadhurNeural",
        "rate": "+10%",
        "pitch": "+30Hz",
        "role": "robot cat helper",
        "speaking_style": "friendly explanatory",
        "translate_as": "helpful friend explaining calmly",
        "tts_note": "clear and friendly",
        "voices": {
            "hi": {"voice": "hi-IN-MadhurNeural", "rate": "+10%", "pitch": "+30Hz"},
            "en": {"voice": "en-US-SteffanNeural", "rate": "+6%", "pitch": "+22Hz"},
        },
    },
    "チャイ": {
        "voices": {
            "hi": {"voice": "hi-IN-SwaraNeural", "rate": "+5%", "pitch": "+10Hz"},
            "en": {"voice": "en-US-JennyNeural", "rate": "+4%", "pitch": "+8Hz"},
        },
    },
    "イゼール": {
        "voices": {
            "hi": {"voice": "hi-IN-SwaraNeural", "rate": "+0%", "pitch": "+0Hz"},
            "en": {"voice": "en-US-EmmaMultilingualNeural", "rate": "+0%", "pitch": "+0Hz"},
        },
    },
    "王妃": {
        "voices": {
            "hi": {"voice": "hi-IN-SwaraNeural", "rate": "-2%", "pitch": "-5Hz"},
            "en": {"voice": "en-US-AriaNeural", "rate": "-4%", "pitch": "-6Hz"},
        },
    },
    "ソドロ": {
        "voices": {
            "hi": {"voice": "hi-IN-MadhurNeural", "rate": "-5%", "pitch": "-15Hz"},
            "en": {"voice": "en-US-TonyNeural", "rate": "-6%", "pitch": "-18Hz"},
        },
    },
    "王": {
        "voices": {
            "hi": {"voice": "hi-IN-MadhurNeural", "rate": "-10%", "pitch": "-20Hz"},
            "en": {"voice": "en-US-GuyNeural", "rate": "-12%", "pitch": "-24Hz"},
        },
    },
    "パル": {
        "voices": {
            "hi": {"voice": "hi-IN-MadhurNeural", "rate": "+0%", "pitch": "+0Hz"},
            "en": {"voice": "en-US-DavisNeural", "rate": "+0%", "pitch": "+0Hz"},
        },
    },
    "評論家": {
        "voices": {
            "hi": {"voice": "hi-IN-MadhurNeural", "rate": "-12%", "pitch": "-18Hz"},
            "en": {"voice": "en-US-RogerNeural", "rate": "-10%", "pitch": "-16Hz"},
        },
    },
    "司会": {
        "voices": {
            "hi": {"voice": "hi-IN-MadhurNeural", "rate": "-3%", "pitch": "-5Hz"},
            "en": {"voice": "en-US-JasonNeural", "rate": "-2%", "pitch": "-4Hz"},
        },
    },
}

LANGUAGE_DEFAULTS: dict[str, dict[str, dict]] = {
    "hi": {
        "female": {"voice": "hi-IN-SwaraNeural", "rate": "+0%", "pitch": "+0Hz"},
        "male": {"voice": "hi-IN-MadhurNeural", "rate": "+0%", "pitch": "+0Hz"},
        "child_male": {"voice": "hi-IN-MadhurNeural", "rate": "+14%", "pitch": "+26Hz"},
    },
    "en": {
        "female": {"voice": "en-US-JennyNeural", "rate": "+0%", "pitch": "+0Hz"},
        "male": {"voice": "en-US-GuyNeural", "rate": "+0%", "pitch": "+0Hz"},
        "child_male": {"voice": "en-US-DavisNeural", "rate": "+12%", "pitch": "+24Hz"},
    },
}

DEFAULT_FEMALE = LANGUAGE_DEFAULTS["hi"]["female"]
DEFAULT_MALE = LANGUAGE_DEFAULTS["hi"]["male"]
DEFAULT_CHILD_MALE = LANGUAGE_DEFAULTS["hi"]["child_male"]

# Curated lines where literal MT fails on tone (normalized Japanese key -> Hindi).
CURATED_HINDI: dict[str, str] = {
    "マイロほかのことして遊ぶのじゃ": "मायलो~ चलो कुछ और करके खेलते हैं न~!",
    "マイロ～ほかのことして遊ぶのじゃ": "मायलो~ चलो कुछ और करके खेलते हैं न~!",
    "絵ばっか描いてないで遊ぶのじゃ": "बस पेंटिंग छोड़ो न~, मेरे साथ खेलने चलो!",
    "絵ばっか描いてないで遊ぶのじゃ！": "बस पेंटिंग छोड़ो न~, मेरे साथ खेलने चलो!",
    "うん…ちょっと待って": "हmm… थोड़ा रुको न…",
    "うん…もうちょっと": "हmm… बस थोड़ा और…",
    "マイロ～": "मायलो~!",
    "ねえ～": "सुनो न~!",
    "じゃあクレアも何か描こうよきっと楽しいよ…": "तो क्लेयर भी कुछ बनाती है न~, बहुत मज़ा आएगा…",
    "ん～ちょっとちょっとってず～っと描いてるのじゃ": "अरे~ 'थोड़ा' 'थोड़ा' कहकर तुम तो बस पेंटिंग ही करते रहते हो!",
}

CURATED_ENGLISH: dict[str, str] = {
    "マイロほかのことして遊ぶのじゃ": "Milo~ let's go do something else and play!",
    "マイロ～ほかのことして遊ぶのじゃ": "Milo~ let's go do something else and play!",
    "絵ばっか描いてないで遊ぶのじゃ": "Stop painting all the time~ come play with me!",
    "絵ばっか描いてないで遊ぶのじゃ！": "Stop painting all the time~ come play with me!",
    "うん…ちょっと待って": "Mm… wait just a second…",
    "うん…もうちょっと": "Mm… just a little longer…",
    "マイロ～": "Milo~!",
    "ねえ～": "Hey~!",
    "じゃあクレアも何か描こうよきっと楽しいよ…": "Then Claire, you should paint something too~ it'll be fun…",
    "ん～ちょっとちょっとってず～っと描いてるのじゃ": "Ugh~ you keep saying 'just a minute' but you've been painting forever!",
}

_CURATED_BY_LANGUAGE = {
    "hi": CURATED_HINDI,
    "en": CURATED_ENGLISH,
}

_TONE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("playful", re.compile(r"(のじゃ|～|〜|ねえ|ほら|もう)")),
    ("gentle", re.compile(r"(うん…|ちょっと待|…|です|ます)")),
    ("panic", re.compile(r"(たすけ|助け|やば|大変|わー|キャー|はっ)")),
    ("boastful", re.compile(r"(見ろ|すごいだろ|ボク|ぼくの)")),
    ("aggressive", re.compile(r"(てめー|殴|黙|うるさい)")),
    ("polite", re.compile(r"(ください|お願い|ありがと)")),
    ("teasing", re.compile(r"(描いてないで|遊ぶ|からか)")),
]


def normalize_japanese(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\{[^}]+\}", "", text)
    text = re.sub(r"[\(（][^）)]*[）)]", "", text)
    text = re.sub(r"\s+", "", text)
    text = text.replace("～", "~").replace("〜", "~")
    return text.strip()


def _subtitle_visible_text(text: str) -> str:
    """Subtitle line without tags, speaker prefixes, and furigana readings."""
    text = re.sub(r"\{[^}]+\}", "", text)
    text = re.sub(r"^[-－—]\s*", "", text.strip())
    text = re.sub(r"^（[^）]+）", "", text)
    visible = re.sub(r"[\(（][^）)]*[）)]", "", text)
    return visible.strip()


# Roars, grunts, screams, laughs-as-noise — not spoken dialogue.
_VOCAL_SFX = re.compile(
    r"^(?:"
    r"グ+[ォオッ~]*|"
    r"うわ+|わ[あぁ]*[っ~]*|"
    r"はっ|はぁ|あっ|あやっ|"
    r"ぶ[~～]*[っ~]*|"
    r"んぐ+|"
    r"テヤ|"
    r"フフフ|"
    r"イヤッホ|"
    r"ハァ|"
    r"よっ|"
    r"ガル|"
    r"キャー|"
    r"おお+"
    r")+$"
)

# TTS should not speak transliterated vocal sounds from bad MT.
_TRANSLATED_VOCAL = re.compile(
    r"^(?:"
    r"gr+r+[!\.]*|"
    r"gurr+[!\.]*|"
    r"hmm+[\.…\s]*|"
    r"mm+[\.…\s]*|"
    r"ah+[!\.]*|"
    r"oh+[!\.]*|"
    r"ugh+[!\.]*|"
    r"ग्र+|ग्रर्र+|"
    r"हmm+[\.…\s]*|"
    r"हा[!\.~ न]*|"
    r"बू[~!\.]*|"
    r"न्ग्ग+|"
    r"तयाह[!\.]*|"
    r"गूओ+|"
    r"बहुत\s*खूब[!\.]*"
    r")$",
    re.I,
)


def is_vocal_filler(text: str) -> bool:
    """True for non-dialogue vocalizations (roars, grunts, gasps, music hums)."""
    raw = text.strip()
    if not raw:
        return True
    if raw.startswith("♪") or raw in {"♪", "♪～", "♪~"}:
        return True

    visible = _subtitle_visible_text(raw)
    if not visible:
        return True

    # Substantial kanji → real line (ignore furigana-only parens).
    if re.search(r"[\u4e00-\u9fff]{2,}", visible):
        return False

    # Common spoken fragments that must stay dubbed.
    if re.search(
        r"(どこ|だっけ|わかった|ちょっと|遊ぶ|描|じゃあ|これ|おもしろ|待って|バカ|"
        r"マイロ|クレア|ドラえも|発掘|発見|宮殿|文明|絵|怪物|現実|神話)",
        visible,
    ):
        return False

    compact = re.sub(r"[\s　!！?？…~～〜\.。~\-—・]", "", visible)
    if not compact:
        return True

    # Pure katakana roars / impact words.
    if re.fullmatch(r"[ァ-ヴー]+", compact):
        return True

    if _VOCAL_SFX.match(compact):
        return True

    # Short gasp/scream without lexical content.
    if len(compact) <= 8 and not re.search(r"[\u4e00-\u9fff]", visible):
        if re.search(r"(グ|うわ|わ[あぁっ]|はっ|ぶ|んぐ|テヤ|よっ|あっ|あや|フフ|ハァ)", compact):
            return True

    return False


def is_translated_filler(text: str) -> bool:
    cleaned = text.strip().rstrip("।.")
    return bool(_TRANSLATED_VOCAL.match(cleaned))


def should_dub_segment(seg: dict) -> bool:
    if seg.get("skip_dub"):
        return False
    source = seg.get("source_ja") or seg.get("text", "")
    if is_vocal_filler(source):
        return False
    translated = seg.get("text", "").strip()
    if translated and is_translated_filler(translated):
        return False
    return bool(translated.strip()) if "text" in seg else True


def _language_key(language: str) -> str:
    return language.split("-", 1)[0].lower()


_FEMALE_VOICE_MARKERS = ("Swara", "Jenny", "Aria", "Emma", "Ana", "Sara", "Nancy", "Sonia", "Amber")

# Single source of truth for all Edge TTS prosody limits.
VOICE_QUALITY_LIMITS: dict[str, tuple[int, int]] = {
    "pitch_hz": (-30, 32),
    "rate_pct": (-18, 18),
    "volume_pct": (-25, 25),
    "char_pitch_window": (-8, 10),
    "char_rate_window": (-6, 8),
    "f0_pitch_delta": (-6, 8),
    "f0_rate_delta": (-3, 4),
    "tone_pitch_delta": (-6, 6),
    "tone_rate_delta": (-8, 8),
}

# Per-gender pitch ceiling (within global pitch_hz max).
VOICE_PITCH_CEILING = {
    "female": 32,
    "male": 28,
}


def _clamp_int(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def parse_pitch_hz(pitch: str | None) -> int:
    match = re.match(r"([+-]?\d+)Hz", str(pitch or "+0Hz"))
    return int(match.group(1)) if match else 0


def format_pitch_hz(hz: int, *, voice: str = "") -> str:
    lo, hi = VOICE_QUALITY_LIMITS["pitch_hz"]
    if voice:
        hi = min(hi, VOICE_PITCH_CEILING["female" if is_female_voice(voice) else "male"])
    return f"{_clamp_int(hz, lo, hi):+d}Hz"


def parse_rate_pct(rate: str | None) -> int:
    match = re.match(r"([+-]?\d+)%", str(rate or "+0%"))
    return int(match.group(1)) if match else 0


def format_rate_pct(pct: int) -> str:
    lo, hi = VOICE_QUALITY_LIMITS["rate_pct"]
    return f"{_clamp_int(pct, lo, hi):+d}%"


def parse_volume_pct(volume: str | None) -> int:
    match = re.match(r"([+-]?\d+)%", str(volume or "+0%"))
    return int(match.group(1)) if match else 0


def format_volume_pct(pct: int) -> str:
    lo, hi = VOICE_QUALITY_LIMITS["volume_pct"]
    return f"{_clamp_int(pct, lo, hi):+d}%"


def is_female_voice(voice: str) -> bool:
    return any(marker in voice for marker in _FEMALE_VOICE_MARKERS)


def _pitch_ceiling_for_voice(voice: str) -> int:
    lo, hi = VOICE_QUALITY_LIMITS["pitch_hz"]
    cap = VOICE_PITCH_CEILING["female" if is_female_voice(voice) else "male"]
    return min(hi, cap)


def clamp_tts_profile(profile: dict, char_profile: dict | None = None) -> dict:
    """Enforce max/min caps on every Edge TTS voice-quality parameter."""
    result = dict(profile)
    voice = result.get("voice", "")
    pitch_lo, _pitch_hi = VOICE_QUALITY_LIMITS["pitch_hz"]
    pitch_hi = _pitch_ceiling_for_voice(voice)
    pitch_hz = parse_pitch_hz(result.get("pitch"))

    if char_profile:
        base_pitch = parse_pitch_hz(char_profile.get("pitch"))
        win_lo, win_hi = VOICE_QUALITY_LIMITS["char_pitch_window"]
        pitch_hz = _clamp_int(pitch_hz, base_pitch + win_lo, base_pitch + win_hi)

    pitch_hz = _clamp_int(pitch_hz, pitch_lo, pitch_hi)
    result["pitch"] = format_pitch_hz(pitch_hz, voice=voice)

    rate_lo, rate_hi = VOICE_QUALITY_LIMITS["rate_pct"]
    rate_pct = parse_rate_pct(result.get("rate"))
    if char_profile:
        base_rate = parse_rate_pct(char_profile.get("rate"))
        win_lo, win_hi = VOICE_QUALITY_LIMITS["char_rate_window"]
        rate_pct = _clamp_int(rate_pct, base_rate + win_lo, base_rate + win_hi)
    rate_pct = _clamp_int(rate_pct, rate_lo, rate_hi)
    result["rate"] = format_rate_pct(rate_pct)

    vol_lo, vol_hi = VOICE_QUALITY_LIMITS["volume_pct"]
    vol_pct = _clamp_int(parse_volume_pct(result.get("volume")), vol_lo, vol_hi)
    result["volume"] = format_volume_pct(vol_pct)

    return result


def _default_for_language(language: str, kind: str) -> dict:
    lang = _language_key(language)
    return dict(LANGUAGE_DEFAULTS.get(lang, LANGUAGE_DEFAULTS["hi"])[kind])


def _voice_settings_for_language(char: dict, language: str) -> dict:
    lang = _language_key(language)
    voices = char.get("voices", {})
    if lang in voices:
        return dict(voices[lang])
    if "hi" in voices:
        return dict(voices["hi"])
    return {key: char[key] for key in ("voice", "rate", "pitch") if key in char}


def character_profile(speaker: str | None, language: str = "hi") -> dict:
    if not speaker:
        return clamp_tts_profile(_default_for_language(language, "male"))
    if speaker in CHARACTER_PROFILES:
        char = CHARACTER_PROFILES[speaker]
        profile = {key: char[key] for key in _CAST_FIELDS if key in char}
        profile.update(_voice_settings_for_language(char, language))
        profile["language"] = _language_key(language)
        return clamp_tts_profile(profile)
    if any(token in speaker for token in ("女", "妃", "母", "姫")):
        return clamp_tts_profile(_default_for_language(language, "female"))
    if any(token in speaker for token in ("王", "男", "魔", "評論", "司会")):
        return clamp_tts_profile(_default_for_language(language, "male"))
    return clamp_tts_profile(_default_for_language(language, "child_male"))


def infer_tone(japanese: str, speaker: str | None) -> str:
    text = japanese
    scores: dict[str, int] = {}
    for name, pattern in _TONE_PATTERNS:
        if pattern.search(text):
            scores[name] = scores.get(name, 0) + 1

    if speaker == "クレア":
        scores["playful"] = scores.get("playful", 0) + 2
        scores["teasing"] = scores.get("teasing", 0) + 1
    elif speaker == "マイロ":
        scores["gentle"] = scores.get("gentle", 0) + 2
    elif speaker == "のび太":
        scores["panic"] = scores.get("panic", 0) + 1
        scores["gentle"] = scores.get("gentle", 0) + 1
    elif speaker == "しずか":
        scores["polite"] = scores.get("polite", 0) + 2
    elif speaker == "ジャイアン":
        scores["aggressive"] = scores.get("aggressive", 0) + 2
    elif speaker == "スネ夫":
        scores["boastful"] = scores.get("boastful", 0) + 2

    if not scores:
        return "neutral"
    return max(scores, key=scores.get)


# Expected voiced pitch band (Hz) for rejecting bleed from the previous speaker.
_CHARACTER_F0_BAND: dict[str, tuple[float | None, float | None]] = {
    "ジャイアン": (None, 275),
    "のび太": (215, None),
    "スネ夫": (205, None),
    "ドラえもん": (175, 420),
    "しずか": (240, None),
}


def f0_matches_character(median_hz: float | None, speaker: str | None) -> bool:
    if median_hz is None or not speaker:
        return True
    band = _CHARACTER_F0_BAND.get(speaker)
    if not band:
        return True
    lo, hi = band
    if lo is not None and median_hz < lo:
        return False
    if hi is not None and median_hz > hi:
        return False
    return True


def _segment_japanese(seg: dict) -> str:
    return seg.get("source_ja") or seg.get("text") or ""


def infer_speaker_from_text(japanese: str, last_speaker: str | None) -> str | None:
    """Guess speaker from dialogue when SRT omits the name tag."""
    text = japanese.strip()
    if not text:
        return None

    if re.search(r"マイロ[~～]", text):
        return "クレア"
    if "のじゃ" in text and ("遊" in text or "描" in text):
        return "クレア"
    if re.search(r"のじゃ[！!]?$", text.replace(" ", "")):
        return "クレア"
    if re.search(r"^(うん|ん～)", text) or "ちょっと待" in text:
        return "マイロ"

    # Nobita pleading / calling Doraemon — not Doraemon himself.
    if re.search(r"待[（(ま)]*ってよ.*ドラえも", text):
        return "のび太"
    if re.search(r"ドラえも[～~んー]*[!！?？…]*$", re.sub(r"\s+", "", text)):
        return "のび太"

    # Gian: rough slang / bully phrasing.
    if re.search(
        r"(?:入[（(は)]*ってみようぜ|どうでもいい|わかんね|てめ[ーえ]|オラ|"
        r"フフフ|歴史.*勉強.*どうでも)",
        text,
    ):
        return "ジャイアン"

    # Suneo: boastful first-person.
    if re.search(r"(?:ぼく|ぼくん)んち", text):
        return "スネ夫"

    # Someone calling Nobita by name — caller, not Nobita.
    if re.search(r"(?:ちょっと|こら|おい|まったく|いいか)[\s　]*のび太", text):
        return "ジャイアン"
    if re.search(r"のび太(?:くん|ちゃん)[\s　!,！?？]", text) and not re.search(
        r"^（のび太", text
    ):
        return "ドラえもん"

    if re.search(r"^（のび太|[-\s　]（のび太", text):
        return "のび太"
    if re.search(r"^(?:ぼく|僕|俺)(?:は|、|の)", text):
        return "のび太"
    if "しずか" in text and re.search(r"^（しずか", text):
        return "しずか"
    return None


# Only inherit the previous speaker when lines are back-to-back (same breath).
_SPEAKER_INHERIT_GAP_SEC = 0.65


def propagate_speaker(segments: list[dict]) -> list[dict]:
    """Fill missing speaker from dialogue cues and very tight same-speaker runs."""
    result = []
    last_speaker = None
    for seg in segments:
        item = dict(seg)
        japanese = _segment_japanese(item)
        if not item.get("speaker"):
            item["speaker"] = infer_speaker_from_text(japanese, last_speaker)
        if not item.get("speaker") and last_speaker:
            gap = item.get("srt_start", item["start"]) - (
                result[-1].get("srt_end", result[-1]["end"]) if result else 999
            )
            if gap < _SPEAKER_INHERIT_GAP_SEC:
                item["speaker"] = last_speaker
        if item.get("speaker"):
            last_speaker = item["speaker"]
        result.append(item)
    return result


def finalize_segment_speakers(segments: list[dict]) -> list[dict]:
    """Re-resolve speakers before TTS so cached labels pick up improved heuristics."""
    refreshed = []
    last_speaker = None
    for seg in segments:
        item = dict(seg)
        japanese = _segment_japanese(item)
        explicit = item.get("speaker")
        inferred = infer_speaker_from_text(japanese, last_speaker)
        if inferred:
            item["speaker"] = inferred
        elif explicit:
            item["speaker"] = explicit
        elif last_speaker:
            gap = item.get("srt_start", item["start"]) - (
                refreshed[-1].get("srt_end", refreshed[-1]["end"]) if refreshed else 999
            )
            if gap < _SPEAKER_INHERIT_GAP_SEC:
                item["speaker"] = last_speaker
        if item.get("speaker"):
            last_speaker = item["speaker"]
        refreshed.append(item)
    return refreshed


def _curated_lookup(japanese: str, language: str) -> str | None:
    curated = _CURATED_BY_LANGUAGE.get(_language_key(language), {})
    key = normalize_japanese(japanese)
    if key in curated:
        return curated[key]
    for pattern, translated in curated.items():
        norm_pattern = normalize_japanese(pattern)
        if norm_pattern in key or key in norm_pattern:
            return translated
    return None


def _google_translate(text: str, target_language: str) -> str:
    return translate_text(text, target_language, backend="google")


def _machine_translate(
    text: str,
    target_language: str,
    *,
    backend: str = "auto",
) -> str:
    return translate_text(text, target_language, backend=backend)


def refine_hindi(
    hindi: str,
    japanese: str,
    speaker: str | None,
    tone: str,
) -> str:
    text = hindi.strip()
    ja_norm = normalize_japanese(japanese)

    # Claire — playful coaxing, not commands
    if speaker == "クレア" and tone in ("playful", "teasing", "neutral"):
        text = re.sub(
            r"केवल\s*चित्र\s*न\s*बनाएं,?\s*खेल(?:ें|ो)?!?",
            "बस पेंटिंग छोड़ो न~, मेरे साथ खेलने चलो!",
            text,
            flags=re.I,
        )
        text = re.sub(r"न\s*बनाएं", "मत करो न", text, flags=re.I)
        text = re.sub(r"([^!~])खेलें!?", r"\1खेलो न~", text, flags=re.I)
        if "遊" in ja_norm and "खेल" in text and "चलो" not in text:
            text = re.sub(r"खेल(?:ें|ो)?", "खेलने चलो न~", text, flags=re.I)
        if tone in ("playful", "teasing") and not re.search(r"(न~|~|!)$", text):
            text = text.rstrip("।.") + " न~"

    # Milo — soft, hesitant
    if speaker == "マイロ" or tone == "gentle":
        text = text.replace("रुकें", "रुको न…")
        text = text.replace("इंतज़ार करें", "थोड़ा रुको…")
        if "うん" in japanese or "…" in japanese:
            if not text.startswith(("हmm", "हाँ", "umm")):
                text = "हmm… " + text.lstrip()

    # Nobita — worried
    if speaker == "のび太" or tone == "panic":
        text = re.sub(r"^([^!]*)(!)$", r"\1!", text)

    # Shizuka — polite
    if speaker == "しずか" or tone == "polite":
        text = text.replace("तुम ", "आप ")
        text = text.replace("करो", "कीजिए")

    # Gian — rough
    if speaker == "ジャイアン" or tone == "aggressive":
        text = text.replace("कृपया ", "")
        text = text.replace("कीजिए", "कर")

    # Suneo — boastful
    if speaker == "スネ夫" or tone == "boastful":
        if "!" not in text:
            text = text.rstrip(".") + "!"

    return text.strip()


def refine_english(
    english: str,
    japanese: str,
    speaker: str | None,
    tone: str,
) -> str:
    text = english.strip()
    ja_norm = normalize_japanese(japanese)

    if speaker == "クレア" and tone in ("playful", "teasing", "neutral"):
        text = re.sub(
            r"stop painting(?: all the time)?(?: and)? play",
            "stop painting all the time~ come play with me",
            text,
            flags=re.I,
        )
        text = re.sub(r"\bplay\b", "play with me", text, count=1, flags=re.I)
        if tone in ("playful", "teasing") and not re.search(r"(~|!)$", text):
            text = text.rstrip(".") + "~"

    if speaker == "マイロ" or tone == "gentle":
        text = text.replace("Wait.", "Wait just a second…")
        if "うん" in japanese or "…" in japanese:
            if not text.lower().startswith(("mm", "um", "uh")):
                text = "Mm… " + text.lstrip()

    if speaker == "のび太" or tone == "panic":
        if tone == "panic" and not text.endswith("!"):
            text = text.rstrip(".") + "!"

    if speaker == "しずか" or tone == "polite":
        text = re.sub(r"\byou\b", "you", text)
        if "please" not in text.lower() and ("ください" in japanese or "お願い" in japanese):
            text = text.rstrip(".!") + ", please."

    if speaker == "ジャイアン" or tone == "aggressive":
        text = text.replace("please ", "").replace("Please ", "")

    if speaker == "スネ夫" or tone == "boastful":
        if "!" not in text:
            text = text.rstrip(".") + "!"

    if "マイロ" in ja_norm and speaker == "クレア" and "Milo" not in text:
        text = "Milo~ " + text

    return text.strip()


def refine_translation(
    translated: str,
    japanese: str,
    speaker: str | None,
    tone: str,
    language: str,
) -> str:
    if _language_key(language) == "en":
        return refine_english(translated, japanese, speaker, tone)
    return refine_hindi(translated, japanese, speaker, tone)


def contextual_translate(
    japanese: str,
    speaker: str | None,
    target_language: str = "hi",
    *,
    translate_backend: str = "auto",
    translate_model: str | None = None,
) -> tuple[str, str]:
    """
    Return (translated_text, tone).
    Uses curated lines, character tone, then Minimax/Google + refinement.
    """
    tone = infer_tone(japanese, speaker)
    if is_vocal_filler(japanese):
        return "", tone

    curated = _curated_lookup(japanese, target_language)
    if curated:
        return curated, tone

    source = _subtitle_visible_text(japanese) or japanese
    translated = _machine_translate(
        source,
        target_language,
        backend=translate_backend,
    )

    translated = refine_translation(translated, japanese, speaker, tone, target_language)
    return translated, tone


def batch_contextual_translate(
    segments: list[dict],
    target_language: str = "hi",
    *,
    translate_backend: str = "auto",
    translate_model: str | None = None,
    workers: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Fast batch translation with per-line tone refinement.

    Returns (translated_segments, failed_segments).
    Failed segments are not cached and will be retried on the next run.
    """
    curated_results: list[dict] = []
    pending_segments: list[dict] = []
    pending_sources: list[str] = []
    failed: list[dict] = []

    for seg in segments:
        source_ja = seg.get("source_ja") or seg["text"]
        speaker = seg.get("speaker")
        if is_vocal_filler(source_ja):
            continue

        tone = infer_tone(source_ja, speaker)
        curated = _curated_lookup(source_ja, target_language)
        if curated:
            curated_results.append(
                {
                    **{k: v for k, v in seg.items() if k != "text"},
                    "source_ja": source_ja,
                    "tone": tone,
                    "text": curated.strip(),
                }
            )
            continue

        source = _subtitle_visible_text(source_ja) or source_ja
        pending_segments.append(seg)
        pending_sources.append(source)

    machine_results: list[dict] = []
    if pending_sources:
        raw_lines = translate_many(
            pending_sources,
            target_language,
            backend=translate_backend,
            workers=workers,
        )
        for seg, raw in zip(pending_segments, raw_lines):
            source_ja = seg.get("source_ja") or seg["text"]
            speaker = seg.get("speaker")
            tone = infer_tone(source_ja, speaker)
            if not raw.strip():
                failed.append(
                    {
                        **seg,
                        "source_ja": source_ja,
                        "speaker": speaker,
                        "reason": "all_backends_failed",
                    }
                )
                continue
            refined = refine_translation(raw, source_ja, speaker, tone, target_language)
            if not refined.strip():
                failed.append(
                    {
                        **seg,
                        "source_ja": source_ja,
                        "speaker": speaker,
                        "reason": "empty_after_refine",
                    }
                )
                continue
            machine_results.append(
                {
                    **{k: v for k, v in seg.items() if k != "text"},
                    "source_ja": source_ja,
                    "tone": tone,
                    "text": refined.strip(),
                }
            )

    return sorted(curated_results + machine_results, key=lambda item: item["start"]), failed


def merge_tts_with_character(
    profile: dict,
    speaker: str | None,
    tone: str,
    language: str = "hi",
) -> dict:
    """Blend detected pitch profile with fixed character voice traits."""
    char = character_profile(speaker, language)
    merged = {**char, **profile}
    lang = _language_key(language)
    rate_pct = parse_rate_pct(merged.get("rate"))
    pitch_hz = parse_pitch_hz(merged.get("pitch"))
    d_rate_lo, d_rate_hi = VOICE_QUALITY_LIMITS["tone_rate_delta"]
    d_pitch_lo, d_pitch_hi = VOICE_QUALITY_LIMITS["tone_pitch_delta"]

    if speaker == "クレア" and tone in ("playful", "teasing"):
        rate_pct += _clamp_int(4 if lang == "hi" else 3, d_rate_lo, d_rate_hi)
    elif speaker == "マイロ" and tone == "gentle":
        rate_pct += _clamp_int(-2, d_rate_lo, d_rate_hi)
    elif speaker == "のび太" and tone == "panic":
        rate_pct += _clamp_int(4, d_rate_lo, d_rate_hi)
        pitch_hz += _clamp_int(3, d_pitch_lo, d_pitch_hi)
    elif speaker == "ジャイアン":
        if tone == "aggressive":
            rate_pct += _clamp_int(-3, d_rate_lo, d_rate_hi)
            pitch_hz += _clamp_int(-3, d_pitch_lo, d_pitch_hi)
        vol_pct = parse_volume_pct(merged.get("volume"))
        vol_pct += _clamp_int(4, *VOICE_QUALITY_LIMITS["volume_pct"])
        merged["volume"] = format_volume_pct(vol_pct)
    elif speaker == "スネ夫" and tone == "boastful":
        rate_pct += _clamp_int(3, d_rate_lo, d_rate_hi)

    merged["rate"] = format_rate_pct(rate_pct)
    merged["pitch"] = format_pitch_hz(pitch_hz, voice=merged.get("voice", ""))
    merged["language"] = lang
    return clamp_tts_profile(merged, char)
