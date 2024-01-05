# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 20:21:44 2023

@author: Administrator
"""
import os
from PIL import Image
import PIL.ImageOps
from utils import change_background_color, BackGroundColor    

#rename 

#inputdir="C:\workspaces\sam\inputs\model2\model2-image"
inputdir=rf"C:\software\ffmpeg-6.1-full_build\model3-pose\16image"
#outputdir="C:\software\ffmpeg-6.1-full_build\model2-ai-image"
def rename_images(begin_index):
    #for dirname in os.listdir(inputdir):
   
        
        for i, filename in enumerate(os.listdir(inputdir)):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                print("---filename------", filename)
                suffix=filename.split(".")[1] 
                #str2 = f"{i:05d}"
                os.rename(inputdir + "/" + filename, inputdir + "/" + f"{(begin_index+i+1):05d}" + f".{suffix}")
            else:
                continue


if __name__ == "__main__":
    rename_images(0)
    print("---rename success------")