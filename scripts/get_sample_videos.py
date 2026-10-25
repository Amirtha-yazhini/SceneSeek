"""Download 7 short public sample videos (Intel IoT DevKit + OpenCV) into data/videos/.

Pair them with data/ground_truth.json, which was hand-labelled for these files.

    python scripts/get_sample_videos.py
"""
import urllib.request
from pathlib import Path

INTEL = "https://github.com/intel-iot-devkit/sample-videos/raw/master/"
FILES = {
    "car-detection.mp4": INTEL + "car-detection.mp4",
    "face-demographics-walking.mp4": INTEL + "face-demographics-walking.mp4",
    "people-detection.mp4": INTEL + "people-detection.mp4",
    "person-bicycle-car-detection.mp4": INTEL + "person-bicycle-car-detection.mp4",
    "store-aisle-detection.mp4": INTEL + "store-aisle-detection.mp4",
    "worker-zone-detection.mp4": INTEL + "worker-zone-detection.mp4",
    "vtest_pedestrians.avi": "https://github.com/opencv/opencv/raw/4.x/samples/data/vtest.avi",
}

out = Path(__file__).resolve().parent.parent / "data" / "videos"
out.mkdir(parents=True, exist_ok=True)
for name, url in FILES.items():
    dest = out / name
    if dest.exists():
        print("have   ", name)
        continue
    print("fetch  ", name)
    urllib.request.urlretrieve(url, dest)
print("done ->", out)
