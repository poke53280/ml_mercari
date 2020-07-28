



from mp4_frames import get_ready_data_dir

from pathlib import Path
import shutil

# Empty all cluster folders and place files on root with prefix c{cluster}

rdy_data_dir = get_ready_data_dir()

l = rdy_data_dir.iterdir()

l_path = [x for x in l if x.is_dir()]

for path in l_path:
    path.is_dir()

    c_l = [x for x in path.iterdir()]

    # Keep all with v5, npy, pkl

    c_l = [x for x in c_l if "v5" in str(x) or "npy" in str(x) or "pkl" in str(x)]

    for x in c_l:
        iCluster = x.parent.stem.split("_")[1]
        dst_path = rdy_data_dir / f"c_{iCluster}_{str(x.name)}"
        assert x.is_file()
        shutil.move(x, dst_path)




