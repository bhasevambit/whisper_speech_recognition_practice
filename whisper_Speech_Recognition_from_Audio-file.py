import torch
import whisper
from modules.pyaudio_audio_signal_processing_training.modules.get_std_input import \
    get_strings_by_std_input
from modules.save_file import save_recognition_result_to_srt_and_txt_file

if __name__ == '__main__':
    # =================
    # === Main Code ===
    # =================

    # CPUで処理させる場合は，このコメントアウトを外す
    torch.cuda.is_available = lambda: False

    # 分析対象の音声データファイルの指定
    # (標準入力にて指定可能とする)
    print("")
    print("=================================================================")
    print("  [ Please INPUT Audio Data File Name (Full-PATH) ]")
    print("=================================================================")
    print("")
    filepath = get_strings_by_std_input()

    print("")
    print("filepath = ", filepath)
    print("")

    # --- Parameters ---
    # 音声ファイルの言語設定 (ja=日本語)
    lang = "ja"
    # ------------------

    # === モデルサイズの指定 ===
    # -----------------------------------------------------------------------------------
    # Size 	 | Parameters | English-only | Multilingual | Required VRAM | Relative speed
    # tiny 	 |  39 M 	  |   tiny.en 	 |    tiny 	    |     ~1 GB 	|    ~32x
    # base 	 |  74 M 	  |   base.en 	 |    base 	    |     ~1 GB 	|    ~16x
    # small  |  244 M 	  |   small.en 	 |    small 	|     ~2 GB 	|    ~6x
    # medium |  769 M 	  |   medium.en  |	  medium 	|     ~5 GB 	|    ~2x
    # large  |  1550 M 	  |   N/A 	     |    large 	|     ~10 GB 	|     1x
    # -----------------------------------------------------------------------------------
    model = whisper.load_model("small")

    # === Audioデータファイルの読み込み ===
    audio = whisper.load_audio(file=filepath)

    # === 音声認識の実行 ===
    result = model.transcribe(audio, verbose=True, language=lang)
    segments = result["segments"]
    # (出力は"text"か"segments"を選択可能)
    # (タイムスタンプにも対応可能な"segments"を使用)

    # === 音声認識結果の標準出力 ===
    # 標準出力として、id毎に改行して表示
    print("")
    print("--- Speech Recognition Results ---")
    for seg in segments:
        id, start, end, text = [seg[key] for key in ["id", "start", "end", "text"]]
        print(f"{id:03}: {start:5.1f} - {end:5.1f} | {text}")
    print("----------------------------------")
    print("")

    # === 音声認識結果のファイル保存 ===
    save_recognition_result_to_srt_and_txt_file(segments)
