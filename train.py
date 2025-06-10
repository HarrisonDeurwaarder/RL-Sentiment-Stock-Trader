import torch
import pandas as pd
import environment as env
import ohlcv as data


def main() -> None:
    print(data.data.columns.values)
    print(env.Environment('AAPL').step())


if __name__ == '__main__':
    main()