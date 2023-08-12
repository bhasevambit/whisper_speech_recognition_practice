import torch
import whisper
from modules.save_file import save_recognition_result_to_srt_and_txt_file
from submodules.pyaudio_audio_signal_processing_training.modules.get_std_input import \
    get_strings_by_std_input

if __name__ == '__main__':
    # =================
    # === Main Code ===
    # =================

    # 分析対象の音声データファイルの指定
    # (標準入力にて指定可能とする)
    print("")
    print("=================================================================")
    print("  [ Please INPUT Audio Data File Name (Full-PATH) ]")
    print("=================================================================")
    print("")
    audio_file_name = get_strings_by_std_input()

    print("")
    print("filepath = ", audio_file_name)
    print("")

    # --- Parameters ---
    # 音声認識における音声ファイルの言語設定 (ja=日本語)
    lang = "ja"

    # PyTorchにて、CPUの利用を強制
    torch.cuda.is_available = lambda: False   # 直接Falseを渡せないためlambda式で渡している

    # Float-Modeの設定 (True:GPU-Mode(float16) / False:CPU-Mode(float32)
    float_mode = False

    # GPU利用可否チェック
    print("\n--- GPU available check ---")
    print("torch.cuda.is_available() = ", torch.cuda.is_available())
    print("float-Mode [True:GPU-Mode(float16) / False:CPU-Mode(float32)] = ", float_mode)
    print("---------------------------\n")
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
    audio = whisper.load_audio(file=audio_file_name)

    # === 音声認識の実行 ===
    result = model.transcribe(audio, verbose=True, language=lang, fp16=float_mode)
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

    print("=================")
    print("= Main Code END =")
    print("=================\n")
