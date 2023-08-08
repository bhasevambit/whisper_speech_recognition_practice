from submodules.pyaudio_audio_signal_processing_training.modules.audio_stream import (
    audio_stream_start, audio_stream_stop)
from submodules.pyaudio_audio_signal_processing_training.modules.gen_freq_domain_data import \
    gen_freq_domain_data
from submodules.pyaudio_audio_signal_processing_training.modules.gen_quef_domain_data import \
    gen_quef_domain_data
from submodules.pyaudio_audio_signal_processing_training.modules.gen_time_domain_data import \
    gen_time_domain_data
from submodules.pyaudio_audio_signal_processing_training.modules.get_mic_index import \
    get_mic_index
from submodules.pyaudio_audio_signal_processing_training.modules.get_std_input import \
    get_selected_mode_by_std_input
from submodules.pyaudio_audio_signal_processing_training.modules.plot_matplot_graph import (
    gen_graph_figure_for_cepstrum, plot_time_and_quef)
from submodules.pyaudio_audio_signal_processing_training.modules.save_audio_to_wav_file import \
    save_audio_to_wav_file
from submodules.pyaudio_audio_signal_processing_training.modules.save_matplot_graph import \
    save_matplot_graph

if __name__ == '__main__':
    # =================
    # === Main Code ===
    # =================

    # --- Parameters ---
    # 動作モード (0:レコーディングモード / 1:リアルタイムモード)
    # (標準入力にて変更可能とする)
    print("")
    print("=================================================================")
    print("  [ Please INPUT MODE type ]")
    print("")
    print("  0 : Recording MODE")
    print("  1 : Real-Time MODE")
    print("=================================================================")
    print("")
    selected_mode = get_selected_mode_by_std_input(mode_count=2)

    if selected_mode == 0:
        selected_mode_name = "'Recording MODE'"
    else:
        selected_mode_name = "'Real-Time MODE'"
    print("\n - Selected MODE = ", selected_mode_name, " - \n")

    # マイクモード (1:モノラル / 2:ステレオ)
    mic_mode = 1

    # サンプリング周波数[Hz]
    if selected_mode == 0:  # レコーディングモード向け
        samplerate = 44100
    else:                   # リアルタイムモード向け
        samplerate = int(44100 / 4)
    print("\nSampling Frequency[Hz] = ", samplerate)

    # 入力音声ストリームバッファあたりのサンプリングデータ数
    if selected_mode == 0:  # レコーディングモード向け
        frames_per_buffer = 512
    else:                   # リアルタイムモード向け
        frames_per_buffer = 1024 * 8
    print(
        "frames_per_buffer [sampling data count/stream buffer] = ",
        frames_per_buffer,
        "\n"
    )

    # グラフタイプ (0:時間領域波形&周波数特性 / 1:時間領域波形&スペクトログラム)
    graph_type = 0

    # 計測時間[s] / 時間領域波形グラフ X軸表示レンジ[s]
    if selected_mode == 0:  # レコーディングモード向け
        time = 10
        time_range = time
        freq_range = int(44100 / 4) / 2
    else:                   # リアルタイムモード向け
        time = 0  # リアルタイムモードの場合は"0"を設定する
        time_range = ((1 / samplerate) * frames_per_buffer) / 10
        freq_range = samplerate / 2

    # デシベル基準値(最小可聴値 20[μPa]を設定)
    dbref = 2e-5

    # 聴感補正(A特性)の有効(True)/無効(False)設定
    A = False   # ケプストラム導出にあたりA特性補正はOFFとする

    # グラフ保存時のファイル名プレフィックス
    filename_prefix = "time-waveform_and_Cepstrum_"
    # ------------------------

    # === マイクチャンネルを自動取得 ===
    index = get_mic_index()[0]
    print("Use Microphone Index :", index, "\n")

    # === グラフ領域作成 ===
    # (リアルタイムモード向けグラフ描画のためにMain Codeでの生成が必須)
    fig, wave_fig, freq_fig, ceps_fig = gen_graph_figure_for_cepstrum()
    # fig       : 生成したmatplotlib figureインスタンス
    # wave_fig  : 時間領域波形向けmatplotlib Axesインスタンス
    # ceps_fig  : ケプストラム向けmatplotlib Axesインスタンス

    # === Microphone入力音声ストリーム生成 ===
    pa, stream = audio_stream_start(
        index, mic_mode, samplerate, frames_per_buffer)
    # pa        : 生成したpyaudio.PyAudioクラスオブジェクト
    #             (pyaudio.PyAudio object)
    # stream    : 生成したpyaudio.PyAudio.Streamオブジェクト
    #             (pyaudio.PyAudio.Stream object)

    # === 時間領域波形 & ケプストラムプロット ===
    # キーボードインタラプトあるまでループ処理継続
    while True:
        try:
            # === 時間領域波形データ生成 ===
            data_normalized, time_normalized = gen_time_domain_data(
                stream, frames_per_buffer, samplerate, time
            )
            # data_normalized : 時間領域波形データ(正規化済)
            # time_normalized : 時間領域波形データ(正規化済)に対応した時間軸データ

            # === 周波数特性データ生成 ===
            spectrum_normalized, amp_normalized, phase_normalized, freq_normalized = gen_freq_domain_data(
                data_normalized, samplerate, dbref, A)
            # spectrum_normalized   : 正規化後 DFTデータ 1次元配列
            # amp_normalized        : 正規化後 DFTデータ振幅成分 1次元配列
            # phase_normalized      : 正規化後 DFTデータ位相成分 1次元配列
            # freq_normalized       : 正規化後 周波数軸データ 1次元配列

            # === ケプストラムデータ生成 ===
            amp_envelope_normalized, cepstrum_data, cepstrum_data_lpl = gen_quef_domain_data(
                data_normalized, samplerate, dbref)
            # amp_envelope_normalized   : 正規化後 スペクトル包絡データ振幅成分 1次元配列
            # cepstrum_data             : ケプストラムデータ(対数値)[dB] 1次元配列
            # cepstrum_data_lpl         : LPL(=Low-Pass-Lifter)適用後
            # ケプストラムデータ(対数値)[dB] 1次元配列

            # === グラフ表示 ===
            plot_time_and_quef(
                fig,
                wave_fig,
                freq_fig,
                ceps_fig,
                data_normalized,
                time_normalized,
                time_range,
                amp_normalized,
                amp_envelope_normalized,
                freq_normalized,
                freq_range,
                cepstrum_data,
                cepstrum_data_lpl,
                dbref,
                A,
                selected_mode
            )

            if selected_mode == 0:
                # レコーディングモードの場合、While処理を1回で抜ける
                break

        except KeyboardInterrupt:
            # 「ctrl+c」が押下された場合、While処理を抜ける
            break

    if selected_mode == 0:
        # レコーディングモードの場合、音声およびグラフを保存する

        # === レコーディング音声のwavファイル保存 ===
        save_audio_to_wav_file(samplerate, data_normalized)

        # === グラフ保存 ===
        save_matplot_graph(filename_prefix)

    # === Microphone入力音声ストリーム停止 ===
    audio_stream_stop(pa, stream)

    print("=================")
    print("= Main Code END =")
    print("=================\n")
