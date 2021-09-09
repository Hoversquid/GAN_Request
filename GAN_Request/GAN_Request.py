import argparse
import math
from pathlib import Path
import sys
from io import BytesIO
import requests
sys.path.append('./taming-transformers')

from IPython import display
from os import chdir, mkdir, path, getcwd, walk, listdir
from os.path import isfile, isdir, exists, join
from IPython.display import clear_output
from shutil import copyfile
import subprocess
import json
import time

class GAN_Request:
    def __init__(self, Other_Txt_Prompts,
                Other_Img_Prompts,
                Other_noise_seeds,
                Other_noise_weights,
                Output_directory,
                Base_Option, Base_Option_Weight,
                Image_Prompt1, Image_Prompt2, Image_Prompt3,
                Text_Prompt1,Text_Prompt2,Text_Prompt3,
                SizeX, SizeY,
                Noise_Seed_Number, Noise_Weight, Seed,
                Image_Model, CLIP_Model,
                Display_Frequency, Clear_Interval, Train_Iterations,
                Step_Size, Cut_N, Cut_Pow,
                Starting_Frame=None, Ending_Frame=None, Overwrite=False, Only_Save=False,
                Overwritten_Dir=None, Frame_Image=False):

        if not path.exists(Output_directory):
            mkdir(Output_directory)

        prompts = {
        "Other_Txt_Prompts": Other_Txt_Prompts,"Other_Img_Prompts": Other_Img_Prompts,
        "Other_noise_seeds": Other_noise_seeds, "Other_noise_weights": Other_noise_weights,
        "Output_directory":Output_directory, "Base_Option":Base_Option,
        "Base_Option_Weight":Base_Option_Weight,"Image_Prompt1":Image_Prompt1,
        "Image_Prompt2":Image_Prompt2,"Image_Prompt3":Image_Prompt3,
        "Text_Prompt1":Text_Prompt1,"Text_Prompt2":Text_Prompt2,
        "Text_Prompt3":Text_Prompt3,"SizeX":SizeX,"SizeY":SizeY,
        "Noise_Seed_Number":Noise_Seed_Number,"Noise_Weight":Noise_Weight,
        "Seed":Seed, "Image_Model":Image_Model,"CLIP_Model":CLIP_Model,
        "Display_Frequency":Display_Frequency,"Clear_Interval":Clear_Interval,
        "Train_Iterations":Train_Iterations,"Step_Size":Step_Size,"Cut_N":Cut_N,
        "Cut_Pow":Cut_Pow,"Starting_Frame":Starting_Frame,"Ending_Frame":Ending_Frame,
        "Only_Save":Only_Save,"Overwritten_Dir":Overwritten_Dir}

        Txt_Prompts = self.get_prompt_list(Text_Prompt1, Text_Prompt2, Text_Prompt3, Other_Txt_Prompts)
        Img_Prompts = self.get_prompt_list(Image_Prompt1, Image_Prompt2, Image_Prompt3, Other_Img_Prompts)

        # Sets the filename based on the collection of Text_Prompts
        Filename = ""
        name_limit = 40
        for i, prompt in enumerate(Txt_Prompts):
            name_length = name_limit - len(Filename)
            if name_length > 0:
              Filename += prompt[:name_length]
              if len(Filename) + 2 < name_limit and i + 1 < len(Txt_Prompts):
                Filename += "__"

        if Filename == "":
          Filename = "No_Prompts"

        Filename = Filename.replace(" ", "_")

        # if Base_Option exists,
        # set the base directory to its final target or targets.
        if not Base_Option in (None, ""):

            sorted_imgs, txt_files = [], []

            is_frames  = False
            file_batch = False
            if not Frame_Image:
                saved_prompts_dir = path.join(Output_directory, "Saved_Prompts/")
                if not path.exists(saved_prompts_dir):
                    mkdir(saved_prompts_dir)

            # Setting the Base_Option to a directory will run each image and saved prompt text file in order.
            # Skips animated files but will run prompts that contain animated file parameters.
            if isdir(Base_Option):
                Base_Dir = Output_directory

                Base_Dir_name = path.basename(Base_Dir)

                files = [f for f in listdir(saved_prompts_dir) if isfile(join(saved_prompts_dir, f))]
                args_basename = path.basename(Base_Option) + "_directory"

                file_batch = True
                files = [join(Base_Option, f) for f in listdir(Base_Option) if isfile(join(Base_Option, f))]

                # Separates images and text files to be run (will currently combine different image sets)
                # TODO: Separate sets of sorted images like in MLAnimator
                imgs = [f for f in files if path.splitext(f)[1] in ('.png', '.jpg')]
                txt_files = [f for f in files if path.splitext(f)[1] == '.txt']
                sorted_imgs = sorted(imgs, key=lambda f: self.get_file_num(f, len(imgs)))

            # Base_Options that are a path/URL to an animated file are separated into frames and ran individually.
            # Images are trained based on the amount of Train_Iterations.
            elif path.splitext(Base_Option)[1] in ('.mp4', '.gif'):
                Base_Dir = self.get_base_dir(Output_directory, Filename, Overwritten_Dir=Overwritten_Dir)
                Base_Dir_name = path.basename(Base_Dir)
                base_file_name = path.basename(path.splitext(Base_Option)[0])
                args_basename = path.basename(Base_Option) + "_animation"

                is_frames = True
                file_batch = True

            # Each run produces a text file of a JSON string. The file contains the settings for the run from which it was made.
            # Running the text file through the program will use the settings saved in it.
            elif path.splitext(Base_Option)[1] in ('.txt'):

                files = [f for f in listdir(saved_prompts_dir) if isfile(join(saved_prompts_dir, f))]
                args_basename = path.basename(path.splitext(Base_Option)[0]) + "_text"

                # Bad DRY
                args_file_name = self.set_valid_filename(files, saved_prompts_dir, args_basename, ".txt")
                self.write_args_file(Output_directory, path.splitext(args_file_name)[0], prompts)
                if not Only_Save:
                    self.run_saved_settings_file(Base_Option)
                return
            else:
                Base_Dir = self.get_base_dir(Output_directory, Filename, Frame_Image=Frame_Image, Overwritten_Dir=Overwritten_Dir)
                print(f"Selecting Base_Option: {Base_Option}\nUsing Filename: {Filename}\nUsing Base_Dir: {Base_Dir}")
                Base_Dir_name = args_basename = path.basename(Base_Dir)

            self.write_args_file(Output_directory, args_basename, prompts)
            # if Only_Save:
            #     return
            #
            # imgLen = len(sorted_imgs)
            #
            # if file_batch:
            #     if imgLen > 0 and self.Train_Iterations > 0:
            #
            #         start, end = 1, imgLen
            #         # If the option is an animated file, setting the Starting_Frame and Ending_Frame can limit from which frames to train.
            #         # Be sure to use the Overwrite option to make frames if they are going in the same directory as other frame directories.
            #         if is_frames:
            #             try:
            #                 if Starting_Frame and Starting_Frame > 1 and Starting_Frame <= imgLen:
            #                     start = Starting_Frame
            #                 if Ending_Frame and Ending_Frame > 1 and Ending_Frame <= imgLen:
            #                     end = Ending_Frame
            #
            #                 frameAmt = end - start
            #                 if frameAmt < 1:
            #                     start, end = 1, imgLen
            #                     print(f"Out of bounds frame selection, running through all {imgLen} frames.")
            #
            #             except:
            #                 start, end = 1, imgLen
            #                 print(f"Invalid frame selection, running through all {imgLen} frames.")
            #
            #             print(f"start: {start}, end: {end}")
            #
            #         j = start
            #
            #         for img in sorted_imgs[start-1:end-1]:
            #             imgname = path.basename(path.splitext(img)[0])
            #
            #             if is_frames:
            #                 target_dir = path.join(Base_Dir, f"{Base_Dir_name}_frame_{j}")
            #             else:
            #                 target_dir = self.get_base_dir(Output_directory, imgname)
            #
            #             j += 1
            #             print(f"Going to target_dir: {target_dir}")
            #
            #             vqgan = VQGAN_CLIP_Z_Quantize(Other_Txt_Prompts,Other_Img_Prompts,
            #                         Other_noise_seeds,Other_noise_weights,target_dir,
            #                         img, Base_Option_Weight,Image_Prompt1,Image_Prompt2,Image_Prompt3,
            #                         Text_Prompt1,Text_Prompt2,Text_Prompt3,SizeX,SizeY,
            #                         Noise_Seed_Number,Noise_Weight,Seed,Image_Model,CLIP_Model,
            #                         Display_Frequency,Clear_Interval,self.Train_Iterations,Step_Size,Cut_N,Cut_Pow,
            #                         Starting_Frame,Ending_Frame,Overwrite,Only_Save,Frame_Image=True)
            #
            #             if is_frames:
            #                 final_dir = path.join(Base_Dir, f"{Base_Dir_name}_final_frames")
            #                 print(f"Copying last frame to {final_dir}")
            #                 if not exists(final_dir):
            #                     mkdir(final_dir)
            #
            #                 files = [f for f in listdir(final_dir) if isfile(join(final_dir, f))]
            #                 seq_num = int(len(files))+1
            #                 sequence_number_left_padded = str(seq_num).zfill(6)
            #                 newname = f"{Base_Dir_name}.{sequence_number_left_padded}.png"
            #                 final_out = path.join(final_dir, newname)
            #                 copyfile(vqgan.final_frame_path, final_out)
            #
            #     if len(txt_files) > 0:
            #         for txt_path in txt_files:
            #             self.run_saved_settings_file(txt_path)
            #     return
            # else:
            #     self.write_args_file(Output_directory, Filename, prompts)
        else:
            self.write_args_file(Output_directory, Filename, prompts)

    def get_base_dir(self, Output_directory, Filename, Frame_Image=False, Overwritten_Dir=None):
        make_unique_dir = True

        # If the rendered file is part of a batch of runs, place file in provided path
        if Frame_Image:
            Base_Dir, make_unique_dir = Output_directory, False

        # Overwritten_Dir is used to place files in a directory that already exists
        elif Overwritten_Dir:
            if not path.exists(Overwritten_Dir):
                print("Directory to overwrite doesn't exist, creating new directory to avoid overwriting unintended directory.")

            else:
                Base_Dir = join(Output_directory, path.basename(Overwritten_Dir))
                make_unique_dir = False

        # Not overwriting will make the filename unique and make a new directory for its files.
        if make_unique_dir: Base_Dir = self.make_unique_dir(Output_directory, Filename)

        return Base_Dir

    def make_unique_dir(self, Output_directory, Filename):
        dirs = [x[0] for x in walk(Output_directory)]
        return self.set_valid_dirname(dirs, Output_directory, Filename)

    def get_file_num(self, f, lastnum):
        namestr = f.split(".")
        if namestr[-2].isnumeric():
            return int(namestr[-2])
        return 0

    # Used to set image path if it's a URL
    def get_pil_imagepath(self, imgpath):
        if imgpath and not path.exists(imgpath):
          imgpath = requests.get(imgpath, stream=True).raw
        return imgpath

    def write_args_file(self, out, base, prompts):
        saved_prompts_dir = path.join(out, "Saved_Prompts/")
        if not path.exists(saved_prompts_dir):
            mkdir(saved_prompts_dir)

        # TODO: change this to CSV, JSON, or XML
        self.filelistpath = saved_prompts_dir + base + ".txt"
        self.write_arg_list(prompts)


    def set_valid_dirname(self, dirs, out, basename, i=0):
        if i > 0:
            newname = "%s(%d)" % (basename, i)
        else:
            newname = basename

        unique_dir_name = True

        if len(dirs) < 1:
            new_path = path.join(out, newname)
            mkdir(new_path)
            return new_path

        for dir in dirs:
            if path.basename(dir) == newname:
                unique_dir_name = False
                break

        if unique_dir_name:
            new_path = path.join(out, newname)

            mkdir(new_path)
            return new_path

        return self.set_valid_dirname(dirs, out, basename, i + 1)


    def set_valid_filename(self, files, out, basename, ext, i=0):
        if i > 0:
            newname = "%s(%d)%s" % (basename, i, ext)
        else:
            newname = "%s%s" % (basename, ext)

        unique_file_name = True

        if len(files) < 1:
            return newname

        for file in files:
            # print(f"checking: {path.basename(file)} against: {newname}")
            if path.basename(file) == newname:

                unique_file_name = False
                break

        if unique_file_name:
            return newname

        return self.set_valid_filename(files, out, basename, ext, i + 1)

    def get_prompt_list(self, first, second, third, rest):
      param_list = [first, second, third]
      param_list = [p for p in param_list if p]
      prompt_list = param_list + rest
      return prompt_list

    def write_arg_list(self,args):
        with open(self.filelistpath, "w", encoding="utf-8") as txtfile:
            json_args = json.dumps(args)
            print(f"writing settings to {self.filelistpath}")
            txtfile.write(json_args)

    def parse_prompt(self, prompt):
        vals = prompt.rsplit('|', 2)
        vals = vals + ['', '1', '-inf'][len(vals):]
        print(f"Vals: {vals}")
        return vals[0], float(vals[1]), float(vals[2])
