import re
from typing import Iterable, List, Optional, Tuple

_HANGUL_BASE = 0xAC00
_HANGUL_LAST = 0xD7A3

_CHOSUNG = ["ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
_JUNGSUNG = ["ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"]
_JONGSUNG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]

_QUALIFIER_PATTERN = re.compile(r"\s*\([^)]*\)\s*$")

# CSV의 여러 장소명에 공통으로 붙는 회사명 접두어. 더 구체적인(긴) 접두어를 먼저
# 검사해야 "에이텍 모빌리티 XXX"에서 "에이텍"만 떼고 "모빌리티 XXX"가 남는 걸 방지한다.
_COMPANY_PREFIXES = ("에이텍 모빌리티", "에이텍 오토", "에이텍 컴퓨터", "에이텍")


def _strip_company_prefix(name: str) -> Optional[str]:
    """"에이텍 회장실", "에이텍 모빌리티 대표이사"처럼 앞에 붙은 회사명을 뗀 나머지를
    반환한다(뗄 접두어가 없거나 나머지가 없으면 None). 이 접두어까지 포함한 전체
    문자열로만 비교하면, 사용자가 회사명 없이 짧게 말한 질문(예: "회장실 어디야")이
    그만큼 자모 길이가 길어진 후보 쪽에 불리하게 작용한다."""
    for prefix in _COMPANY_PREFIXES:
        if name.startswith(prefix + " "):
            remainder = name[len(prefix):].strip()
            return remainder or None
    return None


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
            if not core:
                continue
            # "대표이사 / AFC사업부"처럼 "/"로 여러 개념이 합쳐진 이름을 통째로 한 후보로
            # 넣으면, 합쳐진 만큼 자모 길이(n)가 길어져서 상대거리 계산에서 불리해진다
            # (실제로 이 후보가 전혀 무관한 짧은 후보("랩실")에게 밀리는 버그로 확인됨).
            # "/" 기준으로 쪼개 각각을 독립된 후보로 등록해, 서로 경쟁에서 불이익을 받지
            # 않게 한다.
            for part in core.split("/"):
                part = part.strip()
                if not part:
                    continue
                # 회사명 접두어를 포함한 원본과, 뗀 나머지 둘 다 후보로 등록한다.
                for candidate_text in filter(None, (part, _strip_company_prefix(part))):
                    if candidate_text in seen:
                        continue
                    seen.add(candidate_text)
                    candidate_jamo, _ = _decompose_with_index(candidate_text)
                    if candidate_jamo:
                        self._candidates.append((candidate_text, candidate_jamo))

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
