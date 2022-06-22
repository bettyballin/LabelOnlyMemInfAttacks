import zipfile

def unzip(filename):
    with zipfile.ZipFile(filename+".zip", 'r') as zip_ref:
        zip_ref.extractall(filename)

filename = input("Enter the filename: ")
unzip(filename)