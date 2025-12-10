# stt

音声認識(STT)機能を提供するHTTPサーバーおよびクライアントライブラリ。

## HTTPサーバーの起動

`uvx`を使用し、サーバーを起動。

```bash
uvx --python=3.12 --from 'stt[server] @ git+https://github.com/0xNOY/stt' stt serve http
```

## クライアントとしての使用

サーバー機能を含まない、軽量な依存関係での利用。

### インストール

```bash
uv add "stt @ git+https://github.com/0xNOY/stt"
```

### 使用方法

サーバを起動した状態で`stt`モジュールの`request_stt`関数を使用。

```python
from stt import request_stt

# 音声認識リクエストの送信
text = request_stt()
print(text)
```
