from src.kiosk_pipeline import KioskPipeline


def main():
    pipeline = KioskPipeline()

    print("장소 안내 키오스크 텍스트 조회 테스트 (종료: exit)")
    while True:
        query = input("\n질문> ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue

        answer = pipeline.answer(query)
        pipeline.speak(answer)


if __name__ == "__main__":
    main()
