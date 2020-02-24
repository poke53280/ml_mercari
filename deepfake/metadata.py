

import os
import pathlib
import json

####################################################################################
#
#   get_part_dir
#

def get_part_dir(iPart):


    isLocal = os.name == 'nt'

    if isLocal:
        input_dir = pathlib.Path(f"C:\\Users\\T149900\\Downloads")
        s = input_dir / f"dfdc_train_part_{iPart:02}" / f"dfdc_train_part_{iPart}"
        
    else:
        input_dir = pathlib.Path("/mnt/disks/tmp_mnt/data")
        s = input_dir / f"dfdc_train_part_{iPart}"

    if s.is_dir():
        pass
    else:
        print(str(s))
        assert s.is_dir(), f"{s} not a directory"

    return s


####################################################################################
#
#   read_metadata
#

def read_metadata(iPart):
    
    p = get_part_dir(iPart)

    metadata = p / "metadata.json" 

    assert metadata.is_file()

    txt = metadata.read_text()

    txt_parsed = json.loads(txt)

    l_files = list (txt_parsed.keys())

    l_real_files = []
    l_fake_files = []
    l_original_files = []


    for x in l_files:
        zLabel = txt_parsed[x]['label']
        if zLabel == "REAL":
            file_exists = (p/x).is_file()
            if file_exists:
                l_real_files.append(x)
            else:
                print(f"Warning, missing original file {str(x)} in part {iPart}")

        if zLabel == "FAKE":

            original_file = txt_parsed[x]['original']

            file_exists = (p/x).is_file()
            orig_exists = (p/original_file).is_file()

            if file_exists and orig_exists:
                l_fake_files.append(x)
                l_original_files.append(original_file)
            else:
                print(f"Warning, missing original file {str(original_file)} and/or fake file {str(x)} in part {iPart}")

   
    d = {}

    for x in l_original_files:
        d[x] = []

    assert len (l_fake_files) == len (l_original_files)

    t = list (zip (l_original_files, l_fake_files))

    for pair in t:
        assert pair[0] in d
        d[pair[0]].append(pair[1])            
            

    l_keys = list(d.keys())
    l_keys.sort()

    l_d = []

    for x in l_keys:
        l_d.append((x, d[x]))

    return l_d

def create_file_dictionary(iPart, l_d):
    d = {}

    for o in l_d:
        zTrue = o[0]
        l_zFakes = o[1]

        d[f"p_{iPart}_{zTrue}"] = "True"

        for zFake in l_zFakes:
            d[f"p_{iPart}_{zFake}"] = zTrue

    return d



class MetaData:

    def __init__(self, iPart):        
        self.iPart = iPart
        self.l_d = read_metadata(iPart)
        self.d = create_file_dictionary(iPart, self.l_d)

    def originals(self):
        return self.l_d

    








