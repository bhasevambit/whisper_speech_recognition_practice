import torch
import whisper
from modules.save_file import save_recognition_result_to_srt_and_txt_file
from submodules.pyaudio_audio_signal_processing_training.modules.audio_stream import (
    audio_stream_start, audio_stream_stop)
from submodules.pyaudio_audio_signal_processing_training.modules.gen_time_domain_data import \
    gen_time_domain_data
from submodules.pyaudio_audio_signal_processing_training.modules.get_mic_index import \
    get_mic_index
from submodules.pyaudio_audio_signal_processing_training.modules.get_std_input import (
    get_selected_mic_index_by_std_input, get_selected_mode_by_std_input,
    get_strings_by_std_input)
from submodules.pyaudio_audio_signal_processing_training.modules.save_audio_to_wav_file import \
    save_audio_to_wav_file

if __name__ == '__main__':
    # =================
    # === Main Code ===
    # =================

    # --- Parameters ---
    # # 動作モード (0:音声ファイル読込みモード / 1:マイク音声レコーディングモード)
    # # (標準入力にて変更可能とする)
    print("")
    print("=================================================================")
    print("  [ Please INPUT MODE type ]")
    print("")
    print("  0 : Audio-file-DATA Read MODE")
    print("  1 : Mic-Input-Recording MODE")
    print("=================================================================")
    print("")
    selected_mode = get_selected_mode_by_std_input(mode_count=2)
    # selected_mode = 0   # レコーディングモード固定とする

    if selected_mode == 0:
        selected_mode_name = "'Audio-file-DATA Read MODE'"
    else:
        selected_mode_name = "'Mic-Input-Recording MODE'"
    print("\n - Selected MODE = ", selected_mode_name, " - \n")

    # マイクモード (1:モノラル / 2:ステレオ)
    mic_mode = 1

    # サンプリング周波数[Hz]
    samplerate = 44100
    print("\nSampling Frequency[Hz] = ", samplerate)

    # 入力音声ストリームバッファあたりのサンプリングデータ数
    frames_per_buffer = 512

    print(
        "frames_per_buffer [sampling data count/stream buffer] = ",
        frames_per_buffer,
        "\n"
    )

    # グラフタイプ (0:時間領域波形&周波数特性 / 1:時間領域波形&スペクトログラム)
    graph_type = 0

    # 計測時間[s] / 時間領域波形グラフ X軸表示レンジ[s]
    time = 10
    time_range = time
    freq_range = int(44100 / 4) / 2

    # デシベル基準値(最小可聴値 20[μPa]を設定)
    dbref = 2e-5

    # 聴感補正(A特性)の有効(True)/無効(False)設定
    A = False   # ケプストラム導出にあたりA特性補正はOFFとする

    # グラフ保存時のファイル名プレフィックス
    filename_prefix = "time-waveform_and_Cepstrum_"

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

    if selected_mode == 0:
        # ================================
        # === 音声ファイル読込みモード ===
        # ================================

        # 分析対象の音声データファイルの指定
        # (標準入力にて指定可能とする)
        print("")
        print("=================================================================")
        print("  [ Please INPUT Audio Data File Name (Full-PATH) ]")
        print("=================================================================")
        print("")
        audio_file_name = get_strings_by_std_input()
        print("")

    else:
        # ======================================
        # === マイク音声レコーディングモード ===
        # ======================================

        # === マイクチャンネルを自動取得 ===
        # (標準入力にて選択可能とする)
        print("=================================================================")
        print("  [ Please Select Microphone index ]")
        print("=================================================================")
        print("")
        mic_list = get_mic_index()
        selected_index = get_selected_mic_index_by_std_input(mic_list)
        print("\nUse Microphone Index :", selected_index, "\n")

        # === Microphone入力音声ストリーム生成 ===
        pa, stream = audio_stream_start(
            selected_index, mic_mode, samplerate, frames_per_buffer)
        # pa        : 生成したpyaudio.PyAudioクラスオブジェクト
        #             (pyaudio.PyAudio object)
        # stream    : 生成したpyaudio.PyAudio.Streamオブジェクト
        #             (pyaudio.PyAudio.Stream object)

        # === 時間領域波形データ生成 ===
        data_normalized, time_normalized = gen_time_domain_data(
            stream, frames_per_buffer, samplerate, time
        )
        # data_normalized : 時間領域波形データ(正規化済)
        # time_normalized : 時間領域波形データ(正規化済)に対応した時間軸データ

        # === レコーディング音声のwavファイル保存 ===
        audio_file_name = save_audio_to_wav_file(samplerate, data_normalized)

        # === Microphone入力音声ストリーム停止 ===
        audio_stream_stop(pa, stream)

    print("audio_file_name = ", audio_file_name, "\n")

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
