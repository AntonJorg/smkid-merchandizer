# smkid-merchandizer
![S/M-KID logo.coolimagefileformat](https://github.com/AntonJorg/smkid-merchandizer/blob/main/img/logo.png)
Did you miss out on the awesome S/M-KID council hoodies? Did you not get to Netto in time to buy your daily Harboe? Fear not! With the S/M-KID merchandizer you too can look awesome.

This project uses a deep learning based pose estimation model from [this article](https://learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/) to superimpose the famous S/M-KID hoodie (and other items) onto a single person on your webcam. For best results there should only be one person in frame, and that preson should be visible from the slightly below the waist and up.

## Quickstart guide

1. Clone the repository to your local machine
2. Create environment with required libraries from requirements.txt or merchandizer.yml
3. Download the Caffe model weights from [here](http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel)
4. Place the model weights under pose/mpi
5. Run main.py
6. Enjoy!

## Features
### Current
- Toggleable S/M-KID hoodie with logo
- Banger UI

### Planned
- Other council merch? (Do they even have merch?)
- Assorted beverages for your right hand
- Smatso hotkey
- Ice easter egg (shh don't tell anyone)
