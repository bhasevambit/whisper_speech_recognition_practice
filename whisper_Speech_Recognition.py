import os  # ファイル操作用
# 音声認識のresult[segments]を扱うのに便利なSRTのモジュールをインポートします
from datetime import timedelta

import pysrt
import srt
# import torch
import whisper
from srt import Subtitle

# CPUで処理させる場合は，このコメントアウトを外す
# torch.cuda.is_available = lambda: False

# Size 	 | Parameters | English-only | Multilingual | Required VRAM | Relative speed
# tiny 	 |  39 M 	    |   tiny.en 	 |    tiny 	    |     ~1 GB 	  |    ~32x
# base 	 |  74 M 	    |   base.en 	 |    base 	    |     ~1 GB 	  |    ~16x
# small  |  244 M 	  |   small.en 	 |    small 	  |     ~2 GB 	  |    ~6x
# medium |  769 M 	  |   medium.en  |	  medium 	  |     ~5 GB 	  |    ~2x
# large  |  1550 M 	  |   N/A 	     |    large 	  |     ~10 GB 	  |     1x


dirname = "C:\\Users\\jammi\\VScode_Local_GARAGE\\GitHub_Local\\whisper_speech_recognition_training\\wav\\"
filename = "recorded-sound_20230807_205455.wav"

filepath = dirname + filename  # ファイル指定
print("filepath = ", filepath)

lang = "ja"  # 音声ファイルの言語（ja=日本語）
basename = os.path.splitext(os.path.basename(filepath))[0]  # 音声ファイルの名前（拡張子なし）
print("basename = ", basename)

model = whisper.load_model("small")  # モデルサイズの指定(上の表参照）

# audioファイルを読み込む
audio = whisper.load_audio(file=filepath)

# 音声認識
result = model.transcribe(audio, verbose=True, language=lang)

# 結果の出力はtextかsegments。今回はタイムスタンプにも対応可能なようにsegmentsを使う
segments = result["segments"]

# resultの中身
# {'language': 'ja',
#  'segments': [{
#     "id": len(all_segments),←取り出すやつ
#     "seek": seek,
#     "start": start,　←タイムスタンプ：開始時間
#     "end": end,　←タイムスタンプ：終了時間
#     "text": text,　←テキスト
#     "tokens": result.tokens,
#     "temperature": result.temperature,　
#     "avg_logprob": result.avg_logprob,
#     "compression_ratio": result.compression_ratio,
#     "no_speech_prob": result.no_speech_prob,
# },
#  'text': '********'}

subs = []
for data in segments:  # segmentsの中から、id, start, end, textを取り出していく。
    index = data["id"] + 1
    start = data["start"]
    end = data["end"]
    text = data["text"]

    sub = Subtitle(  # SRTモジュールのSubtitle関数を使って、情報を格納していく
        index=1,
        start=timedelta(seconds=timedelta(seconds=start).seconds, microseconds=timedelta(seconds=start).microseconds),
        end=timedelta(seconds=timedelta(seconds=end).seconds, microseconds=timedelta(seconds=end).microseconds),
        content=text,
        proprietary=''
    )

    subs.append(sub)

# 格納した情報をSRTファイルとして書き出す
with open(f"{basename}.srt", mode="w", encoding="utf-8") as f:
    f.write(srt.compose(subs))

# SRTファイルから必要な情報だけ取り出してtxtファイルで保存する
subrip = pysrt.open(f"{basename}.srt")
f_out = open(f"{basename}_speech_recognitioned.txt", mode="w", encoding="utf-8")

# テキスト（IDとタイムスタンプ無し）
for sub in subrip:
    f_out.write(sub.text + '\n')

# タイムスタンプ、テキスト（ID無し）
# for sub in subrip:
#   f_out.write(str(sub.start) + ' --> ' + str(sub.end) + '\n')
#   f_out.write(sub.text + '\n')
