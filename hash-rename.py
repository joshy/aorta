##
# Rename dir entry
##
import os
import hashlib

path = '/Users/irrwitz/github/aorta/ttt/test'
with os.scandir(path) as it:
    for entry in it:
        print(entry)
        dest =  hashlib.md5(entry.name.encode('utf-8')).hexdigest()
        print(entry.path, '->', path + '/' + dest)
        os.rename(entry.path, path + '/' + dest)