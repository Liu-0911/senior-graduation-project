import os
import tqdm
from hashlib import md5
from functools import partial

def copy_dir(sourceDir,  targetDir):
    if sourceDir.find(".svn") > 0:
        return
    for file in tqdm.tqdm(os.listdir(sourceDir),unit='files',desc = sourceDir):
        sourceFile = os.path.join(sourceDir,  file)
        targetFile = os.path.join(targetDir,  file)
        if os.path.isfile(sourceFile):
            if not os.path.exists(targetDir):
                os.makedirs(targetDir)
            if not  os.path.exists(targetFile) or(
                    os.path.exists(targetFile) and (
                    os.path.getsize(targetFile) !=
                    os.path.getsize(sourceFile))):

                g_s = open(sourceFile, 'rb')
                g = g_s.read()
                md5_source = md5(g).hexdigest()
                md5_target = None
                g_s.close()
                checking_time = 5
                while not (md5_source == md5_target and checking_time > 0):
                    with open(sourceFile, 'rb') as s:
                        with open(targetFile, 'wb') as fd:
                            record_size = 1048576
                            records = iter(partial(s.read, record_size), b'')
                            size = int(os.path.getsize(os.path.abspath(sourceFile))/record_size)
                            for data in tqdm.tqdm(records,
                                                  total=size,
                                                  unit='MB',
                                                  desc = sourceFile,
                                                  mininterval=1,
                                                  ncols=80,):
                                fd.write(data)
                    with open(targetFile, 'rb') as h_s:
                        h =h_s.read()
                        md5_target = md5(h).hexdigest()
                    checking_time -=1
        if os.path.isdir(sourceFile):
            First_Directory = False
            copy_dir(sourceFile, targetFile)

source = r'F:\余额'
target = r'F:\1'

copy_dir(source,target)