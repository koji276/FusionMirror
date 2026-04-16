# FusionMirror

FusionMirror は、商談用会話支援システムのローカル実行プロジェクトです。  
このディレクトリは `FusionMirror` 単体のルートを想定し、音声入力、Rチャンネル中心のゲート処理、ローカルWhisper字幕化、Streamlit UI をまとめて扱います。

## プロジェクト構造

`fusion-mirror` のルートには、次のファイルが置かれる前提です。

- `app.py`
  - 音声取得、チャンネル分離、RMSゲート、Whisper、Streamlit UI を実装するコンクリート層
- `config.json`
  - 実行設定と状態を定義する鉄筋層
- `README_config.md`
  - `config.json` 各項目の役割、推奨値、チューニング根拠
- `requirements.txt`
  - Python依存関係

## RCアーキテクチャ宣言

FusionMirror は「鉄筋とコンクリートの分離」を前提に保守します。

- 鉄筋: `config.json`
  - 音声デバイス、対象チャンネル、VAD相当のしきい値、UI設定、状態を保持する
  - 運用ポリシーやチューニング対象を表す
- コンクリート: `app.py`
  - 音声の取得、L/R処理、無音破棄、Whisper推論、UI表示を実行する
  - 鉄筋を読んで動作し、設定変更で制御できる部分をコードに埋め込まない

この分離により、後続の開発者やAIは次の判断を明確にできます。

- 閾値を変えたい: まず `config.json` を触る
- デバイス選択や対象チャンネルを変えたい: まず `config.json` を触る
- アルゴリズムやUIの流れを変えたい: `app.py` を触る
- Streamlit固有の制約やOS依存のエラーを吸収したい: `app.py` 側で封じ込める

## 開発プロトコル

新機能を追加する場合は、次の順序を守ります。

1. まず `config.json` に必要な項目を定義する
2. 次に `README_config.md` に項目の意味と推奨値を書く
3. 最後に `app.py` で処理を実装する

運用ルール:

- 先にコードを書いて、後から設定をつじつま合わせで足さない
- チューニング可能な値は、可能な限り `config.json` に出す
- 環境依存の回避策は `app.py` に閉じ込め、設定ファイルをUI依存の例外で汚さない

## 実行方法

カレントディレクトリは必ず `fusion-mirror` のルートにします。

```powershell
cd C:\Users\kojit\Documents\projects\fusion-mirror
.\.venv\Scripts\python.exe -m streamlit run app.py
```

注意:

- `app.py` と `config.json` は同じディレクトリに置く
- ルート以外から起動すると、設定ファイルや補助ドキュメントの参照がずれる可能性がある

## 環境構築

このプロジェクトでは、ルート直下の `.venv` を共通仮想環境として使います。

```powershell
cd C:\Users\kojit\Documents\projects\fusion-mirror
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 外部依存関係

### ffmpeg

- `openai-whisper` はローカル推論でも `ffmpeg` を前提にする
- `ffmpeg -version` が通る状態にしておく

### オーディオデバイス

- `sounddevice` で入力デバイスを扱う
- ステレオ入力では Left/Right の物理分離をそのまま検証できる
- モノラル入力では内部的に 2ch 化してデバッグできるが、これは実配線分離の代替ではなくテスト用

## チューニングの基本

まず `config.json` を確認し、次の順で調整します。

1. `audio.device`
   - 使用する入力デバイスを合わせる
2. `audio.target_channel_index`
   - 相手音声が Right 以外に来る場合だけ変更する
3. `vad.rms_threshold`
   - 音声が通らない、またはノイズが通りすぎる場合に最初に調整する
4. `vad.min_speech_duration_ms`
   - 短い発話をどれだけ拾うかを調整する
5. `vad.max_silence_duration_ms`
   - セグメントをどこで切るかを調整する

## 現在の配置について

この README は `fusion-mirror` を正しいルートとして読む前提です。  
もし `app.py`、`config.json`、`README_config.md`、`requirements.txt` が別ディレクトリにあるなら、FusionMirror の実体はまだ移設途中です。実行前にルート配置を揃えてください。
