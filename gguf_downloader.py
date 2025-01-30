"""
gguf_downloader.py

usage:
Command description
python gguf_downloader.py repository/model_name_version

Example Command:
python gguf_downloader.py tensorblock/Qwen-encoder-0.5B-GGUF

Optional --save-dir argument
python gguf_downloader.py tensorblock/Qwen-encoder-0.5B-GGUF --save_dir my_downloads
"""

import os
import argparse
from tqdm import tqdm
from huggingface_hub import hf_hub_download, list_repo_files
import fnmatch

class GGUFDownloader:
    def __init__(self, model_repo: str, save_dir: str = "downloads"):
        """
        Initializes the GGUFDownloader.

        :param model_repo: The Hugging Face model repository in the format "organization/repo".
        :param save_dir: The directory where downloaded files will be saved.
        """
        self.model_repo = model_repo
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def download_gguf_files(self, file_patterns: list = ["*.gguf"]):
        """
        Downloads GGUF files matching the specified patterns from the Hugging Face repository.

        :param file_patterns: List of file patterns to download (e.g., ["*.gguf"]).
        :return: List of paths to downloaded files.
        """
        downloaded_files = []

        # Fetch repository file list from Hugging Face
        repo_files = self._list_files_in_repo()

        files_to_download = [file for pattern in file_patterns for file in repo_files if self._matches_pattern(file, pattern)]

        # Download files with progress bar
        for file in tqdm(files_to_download, desc="Downloading files"):
            try:
                local_path = hf_hub_download(repo_id=self.model_repo, filename=file, cache_dir=self.save_dir)
                downloaded_files.append(local_path)
            except Exception as e:
                print(f"Failed to download {file}: {e}")

        return downloaded_files

    def _list_files_in_repo(self):
        """
        Lists all files in the Hugging Face repository.

        :return: List of filenames.
        """
        return list_repo_files(self.model_repo)

    @staticmethod
    def _matches_pattern(filename, pattern):
        """
        Checks if a filename matches a given pattern.

        :param filename: Name of the file to check.
        :param pattern: File pattern to match (e.g., "*.gguf").
        :return: True if the filename matches the pattern, False otherwise.
        """
        return fnmatch.fnmatch(filename, pattern)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download GGUF files from a Hugging Face repository.")
    parser.add_argument("repo", type=str, help="The Hugging Face model repository in the format 'organization/repo'.")
    parser.add_argument("--save_dir", type=str, default="downloads", help="Directory to save downloaded files (default: downloads).")

    args = parser.parse_args()

    downloader = GGUFDownloader(model_repo=args.repo, save_dir=args.save_dir)

    try:
        files = downloader.download_gguf_files(["*.gguf"])
        print("Downloaded files:", files)
    except Exception as e:
        print("An error occurred:", str(e))
