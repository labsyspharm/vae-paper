import boto3
import botocore
import pathlib
import tqdm

bucket = 'lsp-public-data'

base = pathlib.Path(__file__).resolve().parent.parent
print(f"Will download XX GB of files to {base}\n")

client = boto3.client(
    's3',
    config=botocore.config.Config(signature_version=botocore.UNSIGNED),
)
keys = []
next_token = ''
base_kwargs = {
    'Bucket': bucket,
    'Prefix': 'baker-2025-vae',
}

with tqdm.tqdm(desc="scanning for data files", total=43) as pbar:
    while next_token is not None:
        pbar.update()
        kwargs = base_kwargs.copy()
        if next_token != '':
            kwargs.update({'ContinuationToken': next_token})
        results = client.list_objects_v2(**kwargs)
        contents = results.get('Contents')
        for i in contents:
            k = i.get('Key')
            assert k[-1] != '/'
            keys.append(k)
        next_token = results.get('NextContinuationToken')

print("\nDownloading files...")
for i, k in enumerate(keys, 1):
    meta = client.head_object(Bucket=bucket, Key=k)
    length = int(meta.get('ContentLength', 0))
    rel_path = k.split('/', 1)[1]
    dest_path = base / rel_path
    dest_dir = dest_path.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tqdm.tqdm(
        total=length,
        desc=rel_path,
        bar_format="{percentage:.1f}%|{bar:25} | {rate_fmt} | {desc}",
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        client.download_file(bucket, k, str(dest_path), Callback=pbar.update)
