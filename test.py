
import subprocess
class test:
    def __init__(self,model):
        while True:

            if model=="oui":
                print("Model normal ou tiny ?")
                tiny=input()
                if tiny=="tiny":
                    subprocess.call(" python convert.py yolov3-tiny.cfg yolov3-tiny.weights model_data/yolo.h5 ", shell=True)
                    subprocess.call("python videoframe.py", shell=True)
                    break
                elif tiny=="normal":
                    subprocess.call(" python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5 ", shell=True)
                    subprocess.call("python videoframe.py", shell=True)
                    break

            elif model=="non":

                subprocess.call("python videoframe.py", shell=True)
                break
            else:
                print("reponse impossible")

if __name__ == '__main__':
    print("Enregister le model (oui ou non)  ?")
    model = input()
    test(model)