#!/usr/bin/env python3
# main.py

import subprocess
import sys

def main():
    print("Running augmentation...")
    subprocess.run([sys.executable, 'main_aug.py'])
    
    print("Running training...")
    subprocess.run([sys.executable, 'main_train.py'])
    
    print("Complete. Test with: python camera_test.py")

if __name__ == "__main__":
    main()