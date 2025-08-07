import subprocess
import os

def run_command_multiple_times(command_template, num_iterations):
    ratios = [0.3333, 0.6667]
    
    for i in range(num_iterations):
        img1 = i          # e.g., frame1.jpg
        img2 = i + 1       # e.g., frame2.jpg
        print(f"\nFrame pair: frame{img1}.jpg, frame{img2}.jpg")

        for r_index, ratio in enumerate(ratios):
            print(f"  -> Interpolating at ratio {ratio}")
            command = command_template.format(img1, img2, i, i + 1, ratio)
            print(f"Running: {command}")
            result = subprocess.run(command, shell=True, capture_output=True, text=True)

            print("Output:", result.stdout.strip())
            if result.stderr:
                print("Error:", result.stderr.strip())

if __name__ == "__main__":
    input_file_location = input("Input directory: ")
    try:
        lst = os.listdir(input_file_location)
        interpol_count = len(lst)-1
        print(f"Number of iterations: {interpol_count}")
        # run_command_multiple_times("echo Iteration {}", 4,)
        # run_command_multiple_times("python3 drone_imgint.py --img {}/frame_{}.png {}/frame_{}.png --exp=2 --iternum={} --pathchoice={}", interpol_count, filelocation, filelocation2)
        run_command_multiple_times("python3 inference_img.py --img input/frame{}.jpg input/frame{}.jpg --imgnum {} {} --ratio {}", interpol_count)
    except:
        print(f"Directory {input_file_location} does not exist")