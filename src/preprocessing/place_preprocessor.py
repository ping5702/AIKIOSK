from pathlib import Path
from typing import Dict, List

import pandas as pd

# 최소 컬럼 요구사항만 정의해둔다. hours/phone은 데이터에 따라 없을 수 있어 선택 항목으로 둔다.
REQUIRED_COLUMNS = ["id", "name", "category", "description", "location"]


def load_places(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"CSV에 필수 컬럼이 없습니다: {missing}")
    return df


def build_embedding_text(row: Dict[str, str]) -> str:
    parts = [
        f"{row['name']} ({row['category']})",
        row["description"],
        f"위치: {row['location']}",
    ]
    if row.get("hours"):
        parts.append(f"운영시간: {row['hours']}")
    return " / ".join(p for p in parts if p)


def preprocess_places(csv_path: Path) -> List[Dict[str, str]]:
    """CSV를 읽어 각 장소 레코드에 임베딩용 텍스트(embedding_text)를 추가해 반환한다."""
    df = load_places(csv_path)
    records = df.to_dict(orient="records")
    for record in records:
        record["embedding_text"] = build_embedding_text(record)
    return records
