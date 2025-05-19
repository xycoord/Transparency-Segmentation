from multiprocessing import Process, Queue
from queue import Empty
from typing import NamedTuple, List
from PIL import Image

from PIL import PngImagePlugin
import xattr
import traceback
import os
import subprocess
from pathlib import Path


class SaveItem(NamedTuple):
    image: Image.Image
    path: Path
    metadata: dict = None  # Optional metadata

def save_worker(queue: Queue):
    """Worker process that saves images from the queue"""
    while True:
        try:
            item = queue.get(timeout=30)
            if item is None:
                break

            # Save the image first
            item.image.save(item.path)
            path_str = str(item.path)
            
            # Ensure image is written to disk
            with open(path_str, 'rb') as f:
                os.fsync(f.fileno())
            
            if item.metadata:
                try:
                    for k, v in item.metadata.items():
                        attr_name = f'user.{k}'
                        attr_value = str(v)
                        
                        # Use setfattr command which we know works
                        subprocess.run(
                            ['setfattr', '-n', attr_name, '-v', attr_value, path_str], 
                            check=True
                        )
                        
                        # Sync the directory to ensure xattr is written
                        dirfd = os.open(os.path.dirname(path_str), os.O_RDONLY)
                        try:
                            os.fsync(dirfd)
                        finally:
                            os.close(dirfd)
                            
                except Exception as e:
                    print(f"Error setting/verifying xattr: {e}")
                    print(traceback.format_exc())
                
        except Empty:
            continue
        except Exception as e:
            print(f"Error in save worker: {e}")
            print(traceback.format_exc())

# Modify AsyncImageSaver to handle metadata
class AsyncImageSaver:
    def __init__(self, num_workers=2):
        self.queue = Queue(maxsize=100)
        self.workers = []
        
        for _ in range(num_workers):
            p = Process(target=save_worker, args=(self.queue,))
            p.start()
            self.workers.append(p)
    
    def save(self, images: List[Image.Image], paths: List[Path], metadata_list: List[dict] = None):
        """Queue images to be saved with optional metadata"""
        if metadata_list is None:
            metadata_list = [None] * len(images)
            
        for img, path, metadata in zip(images, paths, metadata_list):
            self.queue.put(SaveItem(img, path, metadata))

    def close(self):
        """Stop all workers and wait for queue to empty"""
        # Send sentinel value to each worker
        for _ in self.workers:
            self.queue.put(None)
        
        # Wait for workers to finish
        for p in self.workers:
            p.join()