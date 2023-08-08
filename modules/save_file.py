import datetime
import os

import pysrt
import srt


def save_recognition_result_to_srt_and_txt_file(segments):
    # ========================================
    # === 音声認識結果 srtファイル保存関数 ===
    # ========================================
    # segments : whisperライブラリ 音声認識結果
    # (whisper.load_modelクラスオブジェクトのtranscribe()メソッド出力の"segments"データ)

    print("File Save START")

    now = datetime.datetime.now()

    dirname = 'result/'
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    # 拡張子無しファイル名の設定
    filename = dirname + 'speech_recognition_' + \
        now.strftime('%Y%m%d_%H%M%S')

    # === srtファイルとして音声認識結果を保存 ===
    subs = []

    # segmentsの中から、id, start, end, textを取得
    for data in segments:
        start = data["start"]
        end = data["end"]
        text = data["text"]

        # SRTモジュールのSubtitle関数を用いて情報を格納
        sub = srt.Subtitle(
            index=1,
            start=datetime.timedelta(
                seconds=datetime.timedelta(seconds=start).seconds,
                microseconds=datetime.timedelta(seconds=start).microseconds
            ),
            end=datetime.timedelta(
                seconds=datetime.timedelta(seconds=end).seconds,
                microseconds=datetime.timedelta(seconds=end).microseconds
            ),
            content=text,
            proprietary=''
        )

        subs.append(sub)

    # 格納した情報をSRTファイルとして書き出し
    with open(f"{filename}.srt", mode="w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

    # === SRTファイルから音声認識テキスト結果のみを抽出しtxtファイルに保存 ===
    subrip = pysrt.open(f"{filename}.srt")

    with open(f"{filename}.txt", mode="w", encoding="utf-8") as f_out:

        for sub in subrip:
            f_out.write(sub.text + '\n')

    print("File Save END\n")
