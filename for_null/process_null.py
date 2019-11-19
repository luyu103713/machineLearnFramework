import pandas as pd

def delete_null(df):
    df = df.dropna(axis=0,how='any')
    print('delete ok')

    return df



def main():
    print('start work!')
if __name__ == '__main__':
    main()