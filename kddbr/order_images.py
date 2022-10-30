import pandas as pd
import imagehash
from PIL import Image
import plotly.express as px

from context import DATADIR

def get_hash(row, hash_function=imagehash.dhash, **kwargs):
    base_dir = DATADIR / 'raw'
    base_dir = base_dir / 'test/test' if pd.isna(row['North']) else base_dir / 'train/train'
    
    with Image.open(base_dir / row['Filename']) as img:
        left = hash_function(img.crop([0, 0, 120, 120]), **kwargs)
        right = hash_function(img.crop([120, 0, 240, 120]), **kwargs)

    return pd.Series({'left_hash': left, 'right_hash': right})

from functools import partial


def build_link_dataframe(_df, hash_size=16):    
    from pandarallel import pandarallel
    pandarallel.initialize(progress_bar=True)

    df = _df.copy()
    if df.index.name == 'Filename':
        df = df.reset_index()
        
    hashes = df.parallel_apply(partial(get_hash, hash_size=hash_size), axis=1)
    
    df = df.join(hashes).reset_index()
    for c in ['left_hash', 'right_hash']:
        df[c] = df[c].astype(str)

    return df.reset_index().set_index('right_hash').join(df.reset_index().set_index('left_hash'), rsuffix='_next').sort_index()[['index', 'index_next']]


def build_sequences(link_df):
    ending_frames = link_df.groupby('index')['index_next'].count()[link_df.groupby('index')['index_next'].count() == 0]

    tmp = link_df.set_index('index')
    print(len(ending_frames), 'frames without next frame.')
    sequences = []

    for idx in ending_frames.index:
        seq = [idx]

        if idx not in tmp.index:
            print(f'{idx} already used in another sequence. Skipping')
            continue
        
        frame = idx
        while True:
            tmp = tmp.loc[tmp.index != frame].copy()  # Cannot link to same frame again
            previous_frames = tmp.loc[tmp['index_next'] == frame]
            
            if len(previous_frames) == 0:
                sequences.append(seq[::-1])  # Reverse order
                break

            seq.append(previous_frames.index[-1])
            frame = previous_frames.index[-1]

    return sequences


def evaluate_sequences(sequences, _df, plot=True):
    flattened = []
    seq_index = []

    i = 0
    for seq in sequences:
        if len(seq) > 0:
            flattened.extend(seq)
            seq_index.extend([i] * len(seq))
            i += 1

    df = _df.copy()
    if df.index.name == 'Filename':
        df = df.reset_index()

    print(len(flattened), len(seq_index), pd.Series(flattened).is_unique, max(seq_index))

    not_in_sequences = df[~df.index.isin(flattened)].copy().sort_values('Altitude')
    ordered = pd.concat([df.loc[flattened], not_in_sequences])
    ordered['sequence'] = seq_index + [max(seq_index) + 1] * len(not_in_sequences)
    ordered['sequence'] = ordered['sequence'].astype(str)
    print('Unique indices:', ordered.index.is_unique)
    print('Same length as original:', len(ordered) == len(df))

    if plot:
        fig = px.scatter(ordered, 
            x=list(range(len(ordered))), 
            y='Altitude', 
            color='sequence',
            width=1280,
            height=720)
        fig.show()

    return ordered.set_index('Filename')


def order_from_df(df, hash_size=16, plot=False):
    links = build_link_dataframe(df, hash_size=hash_size)

    seqs = build_sequences(links)

    return evaluate_sequences(seqs, df, plot=plot)


def main(input_path, output_path, hash_size, plot):
    df = pd.read_csv(input_path)
    ordered = order_from_df(df, hash_size=hash_size, plot=plot)

    ordered.to_csv(output_path)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--in_path', required=True, type=str)
    ap.add_argument('-o', '--out_path', required=True, type=str)
    ap.add_argument('-hs', '--hashsize', type=int, default=16)
    ap.add_argument('-p', '--plot', action='store_true')
    args = ap.parse_args()

    main(args.in_path, args.out_path, args.hashsize, args.plot)
