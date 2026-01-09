import shutil, os, stat

def remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def reset_storage():
    for path in ["data/repos", "data/faiss_index"]:
        if os.path.exists(path):
            shutil.rmtree(path, onerror=remove_readonly)
        os.makedirs(path, exist_ok=True)
