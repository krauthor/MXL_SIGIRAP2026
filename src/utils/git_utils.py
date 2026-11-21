import subprocess
import os
from typing import Dict, Optional


def get_git_commit_info() -> Dict[str, Optional[str]]:
    git_info = {
        "commit_hash": None,
        "commit_hash_short": None,
        "branch": None,
        "is_dirty": None,
        "remote_url": None,
        "commit_message": None,
        "commit_author": None,
        "commit_date": None,
    }

    try:
        subprocess.check_output(
            ["git", "rev-parse", "--git-dir"],
            stderr=subprocess.DEVNULL,
            cwd=os.getcwd()
        )

        try:
            result = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=os.getcwd()
            )
            git_info["commit_hash"] = result.decode().strip()
        except subprocess.CalledProcessError:
            pass

        try:
            result = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=os.getcwd()
            )
            git_info["commit_hash_short"] = result.decode().strip()
        except subprocess.CalledProcessError:
            pass

        try:
            result = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=os.getcwd()
            )
            git_info["branch"] = result.decode().strip()
        except subprocess.CalledProcessError:
            pass

        try:
            result = subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
                cwd=os.getcwd()
            )
            git_info["is_dirty"] = len(result.decode().strip()) > 0
        except subprocess.CalledProcessError:
            pass

        try:
            result = subprocess.check_output(
                ["git", "config", "--get", "remote.origin.url"],
                stderr=subprocess.DEVNULL,
                cwd=os.getcwd()
            )
            git_info["remote_url"] = result.decode().strip()
        except subprocess.CalledProcessError:
            pass

        try:
            result = subprocess.check_output(
                ["git", "log", "-1", "--pretty=format:%s"],
                stderr=subprocess.DEVNULL,
                cwd=os.getcwd()
            )
            git_info["commit_message"] = result.decode().strip()
        except subprocess.CalledProcessError:
            pass

        try:
            result = subprocess.check_output(
                ["git", "log", "-1", "--pretty=format:%an"],
                stderr=subprocess.DEVNULL,
                cwd=os.getcwd()
            )
            git_info["commit_author"] = result.decode().strip()
        except subprocess.CalledProcessError:
            pass

        try:
            result = subprocess.check_output(
                ["git", "log", "-1", "--pretty=format:%ci"],
                stderr=subprocess.DEVNULL,
                cwd=os.getcwd()
            )
            git_info["commit_date"] = result.decode().strip()
        except subprocess.CalledProcessError:
            pass

    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return git_info


def print_git_info(git_info: Dict[str, Optional[str]]) -> None:
    print("------------------ Git Information ------------------")

    if git_info["commit_hash"]:
        print(f"Git Commit Hash: {git_info['commit_hash']}")
        if git_info["commit_hash_short"]:
            print(f"Git Commit Hash (Short): {git_info['commit_hash_short']}")
    else:
        print("Git Commit Hash: Not available (not a git repository or git not installed)")

    if git_info["branch"]:
        print(f"Git Branch: {git_info['branch']}")

    if git_info["is_dirty"] is not None:
        status = "Yes (uncommitted changes)" if git_info["is_dirty"] else "No"
        print(f"Repository Dirty: {status}")

    if git_info["commit_message"]:
        print(f"Commit Message: {git_info['commit_message']}")

    if git_info["commit_author"]:
        print(f"Commit Author: {git_info['commit_author']}")

    if git_info["commit_date"]:
        print(f"Commit Date: {git_info['commit_date']}")

    if git_info["remote_url"]:
        print(f"Remote URL: {git_info['remote_url']}")

    print("----------------------------------------------------")
