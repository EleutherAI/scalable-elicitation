import subprocess
import sys


def run_command(line_number):
    try:
        with open("/home/fslcollab366/byu_cmds.sh", "r") as file:
            lines = file.readlines()
            if 1 <= line_number <= len(lines):
                command = lines[line_number - 1].strip()
                subprocess.run(command, shell=True, check=True)
            else:
                print(f"Error: Line number {line_number} is out of range.")
    except FileNotFoundError:
        print("Error: byu_cmds.sh file not found.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_byu.py <line_number>")
    else:
        try:
            line_number = int(sys.argv[1]) + 1
            run_command(line_number)
        except ValueError:
            print("Error: Please provide a valid integer for the line number.")
