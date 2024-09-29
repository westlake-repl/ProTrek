import os
import pandas as pd
import copy


# efficiently read large file by using a pointer
class FileReader:
    @classmethod
    # build a pointer for file
    def build_pointer(cls, file_path, save_path):
        with open(save_path, 'w') as w, open(file_path, 'r') as r:
            loc = r.tell()
            line = r.readline()

            # record the index of the beginning position for each line
            cnt = 0
            while line:
                w.write(f'{loc}\n')
                loc = r.tell()
                line = r.readline()

                cnt += 1
                if cnt % 100000 == 0:
                    print(f"\rAlready loaded {cnt} rows", end='')

    def __init__(self, file_path):
        # If the pointer doesn't exist, build it
        pointer_path = f"{file_path}.pointer"
        if not os.path.exists(pointer_path):
            FileReader.build_pointer(file_path, pointer_path)

        self.file = open(file_path, 'r')
        self.pointer = []
        with open(pointer_path, 'r') as r:
            for line in r:
                self.pointer.append(int(line))

    # return specific line given index
    def get(self, index):
        loc = self.pointer[index]
        self.file.seek(loc)
        return self.file.readline()[:-1]

    def __len__(self):
        return self.pointer.shape[0]
    
    
def get_file_readers(file_path: str, n_readers: int) -> list:
    """
    This function is used to read large file by using multiple readers
    Args:
        file_path: path to file
        n_readers: number of readers

    Returns: a list of file readers

    """
    reader = FileReader(file_path)
    reader.file.close()
    reader_list = []
    for i in range(n_readers):
        tmp = copy.copy(reader)
        tmp.file = open(file_path, "r")
        reader_list.append(tmp)
    
    return reader_list
