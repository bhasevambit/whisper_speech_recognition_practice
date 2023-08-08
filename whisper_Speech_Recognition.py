import os

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

    lang = "ja"  # 音声ファイルの言語（ja=日本語）
    basename = os.path.splitext(os.path.basename(filepath))[0]  # 音声ファイルの名前（拡張子なし）

    print("")
    print("filepath = ", filepath)
    print("basename = ", basename)
    print("")

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

    # === 音声認識結果出力 ===
    # (出力は"text"か"segments"を選択可能)
    # (タイムスタンプにも対応可能なようにsegmentsを使用)
    segments = result["segments"]

    # 標準出力として、id毎に改行して表示
    print("")
    print("--- Speech Recognition Results ---")
    for seg in result["segments"]:
        id, start, end, text = [seg[key] for key in ["id", "start", "end", "text"]]
        print(f"{id:03}: {start:5.1f} - {end:5.1f} | {text}")
    print("----------------------------------")

    # # segmentsの中から、id, start, end, textを取得
    # subs = []
    # for data in segments:
    #     # index = data["id"] + 1
    #     start = data["start"]
    #     end = data["end"]
    #     text = data["text"]

    #     # SRTモジュールのSubtitle関数を用いて情報を格納
    #     sub = Subtitle(
    #         index=1,
    #         start=timedelta(
    #             seconds=timedelta(
    #                 seconds=start).seconds, microseconds=timedelta(
    #                 seconds=start).microseconds),
    #         end=timedelta(seconds=timedelta(seconds=end).seconds, microseconds=timedelta(seconds=end).microseconds),
    #         content=text,
    #         proprietary=''
    #     )

    #     subs.append(sub)

    # # 格納した情報をSRTファイルとして書き出し
    # with open(f"{basename}.srt", mode="w", encoding="utf-8") as f:
    #     f.write(srt.compose(subs))

    # # SRTファイルから必要な情報だけ取り出してtxtファイルに保存
    # subrip = pysrt.open(f"{basename}.srt")

    # print("subrip = ", subrip)

    # with open(f"{basename}.txt", mode="w", encoding="utf-8") as f_out:

    #     for sub in subrip:
    #         print("sub.text = ", sub.text)
    #         f_out.write(sub.text + '\n')

    save_recognition_result_to_srt_and_txt_file(segments)
