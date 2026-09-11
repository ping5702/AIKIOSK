import re
from typing import Iterable, List, Optional, Tuple

_HANGUL_BASE = 0xAC00
_HANGUL_LAST = 0xD7A3

_CHOSUNG = ["ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
_JUNGSUNG = ["ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"]
_JONGSUNG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]

_QUALIFIER_PATTERN = re.compile(r"\s*\([^)]*\)\s*$")


def _decompose_with_index(text: str) -> Tuple[List[str], List[int]]:
    """텍스트를 자모(초성/중성/종성) 시퀀스로 분해한다. 각 자모가 원문의 몇 번째
    글자에서 나왔는지(jamo -> char index)도 같이 반환해, 나중에 자모 단위로 찾은
    매칭 구간을 원문 글자 구간으로 되돌릴 수 있게 한다. 한글이 아닌 문자는 그대로 통과시킨다."""
    jamo: List[str] = []
    index: List[int] = []
    for i, ch in enumerate(text):
        code = ord(ch) - _HANGUL_BASE
        if 0 <= code <= _HANGUL_LAST - _HANGUL_BASE:
            cho, rem = divmod(code, 588)
            jung, jong = divmod(rem, 28)
            jamo.append(_CHOSUNG[cho])
            index.append(i)
            jamo.append(_JUNGSUNG[jung])
            index.append(i)
            if jong:
                jamo.append(_JONGSUNG[jong])
                index.append(i)
        else:
            jamo.append(ch)
            index.append(i)
    return jamo, index


def _edit_distance(a: List[str], b: List[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def _strip_qualifier(name: str) -> str:
    """"임원실 (4층 좌측)" 같은 층/위치 구분용 괄호는 실제 발화에 등장하지 않으므로
    매칭 대상에서 제외한다."""
    return _QUALIFIER_PATTERN.sub("", name).strip()


class PlaceNameCorrector:
    """STT가 발음이 비슷한 다른 글자로 잘못 인식한 장소명(예: '해장실' -> '회장실')을,
    알려진 장소명 목록과 자모(초성/중성/종성) 단위 편집거리로 비교해 가장 가까운 이름으로
    치환한다. 장소명이 미리 정해진 폐쇄형 어휘일 때만 신뢰할 수 있는 방식이라, BiEncoder
    임베딩이나 LLM 없이도 순수 문자열 연산만으로 동작해 지연이 거의 없다."""

    def __init__(self, place_names: Iterable[str], max_relative_distance: float = 0.3):
        self.max_relative_distance = max_relative_distance
        self._candidates: List[Tuple[str, List[str]]] = []
        seen = set()
        for name in place_names:
            core = _strip_qualifier(name)
            if not core or core in seen:
                continue
            seen.add(core)
            core_jamo, _ = _decompose_with_index(core)
            if core_jamo:
                self._candidates.append((core, core_jamo))

    def correct(self, text: str) -> str:
        if not text or not self._candidates:
            return text

        query_jamo, query_index = _decompose_with_index(text)

        # 후보 이름마다 자모 길이가 다르므로, 절대 편집거리로 비교하면 짧은 이름이
        # 구조적으로 유리해진다(차이가 날 수 있는 자리 자체가 적으므로). 자모 길이 대비
        # 상대 거리로 비교해야 "화상회의실"처럼 긴 이름도 공정하게 경쟁할 수 있다.
        best: Optional[Tuple[float, int, str, int, int]] = None  # (relative_dist, dist, name, start, end)
        for name, name_jamo in self._candidates:
            n = len(name_jamo)
            for window_len in range(max(1, n - 1), n + 2):
                if window_len > len(query_jamo):
                    continue
                for start in range(0, len(query_jamo) - window_len + 1):
                    window = query_jamo[start : start + window_len]
                    dist = _edit_distance(window, name_jamo)
                    relative_dist = dist / n
                    if best is None or relative_dist < best[0]:
                        best = (relative_dist, dist, name, start, start + window_len - 1)

        if best is None:
            return text

        relative_dist, _dist, name, start, end = best
        if relative_dist > self.max_relative_distance:
            return text

        start_char = query_index[start]
        end_char = query_index[end] + 1
        return text[:start_char] + name + text[end_char:]
