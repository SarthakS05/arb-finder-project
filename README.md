# Arbitrage Finder Prototypes

An educational collection of PyTorch, computer-vision, BitTorrent, and cryptocurrency statistical-arbitrage components.

## Included components

- Budget-forced generation wrapper and small BERT-style encoder/classification models
- OpenCV chessboard-camera demonstration
- BitTorrent handshake and piece-request primitives
- Minute-price dataset and a random-forest statistical-arbitrage backtest helper

This is prototype code, not a complete application. `main()` is empty, so running the script does not execute the helpers. No market data is bundled or fetched by this script. The backtest helper expects caller-provided price DataFrames containing `timestamp` and `close`; timestamps are interpreted as Unix seconds when building the equity series. The BitTorrent code does not implement torrent parsing or a complete download client.

## Requirements

- Python 3.9 or newer
- Dependencies listed in `requirements.txt`

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

Import the components from `arb_finder.py` and provide your own data and hardware resources as needed. This repository is for educational use and does not provide financial advice.