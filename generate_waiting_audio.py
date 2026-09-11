import config
from src.tts.tts_client import MeloTTSClient


def main():
    client = MeloTTSClient(host=config.TTS_SERVER_HOST)
    audio_bytes = client.synthesize(config.WAITING_MESSAGE)

    config.AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.WAITING_AUDIO_PATH.write_bytes(audio_bytes)
    print(f"대기 안내음을 생성했습니다. -> {config.WAITING_AUDIO_PATH}")


if __name__ == "__main__":
    main()
