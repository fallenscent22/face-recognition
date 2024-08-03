import os

path = 'C:\\Users\\Nikhitha Chatla\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python311\\site-packages\\torch\\include\\ATen\\ops\\'

if os.path.exists(path):
    print("Path exists.")
else:
    print("Path does not exist.")
