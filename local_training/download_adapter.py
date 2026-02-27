"""Download MI LoRA adapter from Tinker."""

import argparse
import os
import tarfile
import urllib.request

import tinker

CHECKPOINT_PATH = (
    "tinker://6276618e-3bcf-5219-95b3-9de7e9918e23:train:0"
    "/sampler_weights/qwen_robust_plpd_epoch_10"
)
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "adapter")


def main():
    parser = argparse.ArgumentParser(description="Download MI LoRA adapter from Tinker")
    parser.add_argument(
        "--checkpoint-path", default=CHECKPOINT_PATH,
        help="Tinker checkpoint path to download",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help="Directory to save adapter files",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    sc = tinker.ServiceClient()
    rc = sc.create_rest_client()

    print(f"Getting download URL for: {args.checkpoint_path}")
    response = rc.get_checkpoint_archive_url_from_tinker_path(
        args.checkpoint_path
    ).result()

    archive_path = os.path.join(args.output_dir, "archive.tar")
    print(f"Downloading to {archive_path}...")
    urllib.request.urlretrieve(response.url, archive_path)

    print(f"Extracting to {args.output_dir}...")
    with tarfile.open(archive_path) as tar:
        tar.extractall(args.output_dir)
    os.remove(archive_path)

    # Verify expected files exist
    expected = ["adapter_model.safetensors", "adapter_config.json"]
    for fname in expected:
        fpath = os.path.join(args.output_dir, fname)
        if os.path.exists(fpath):
            print(f"  Found: {fname}")
        else:
            print(f"  WARNING: Missing {fname}")

    print(f"Adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
