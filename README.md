# Bachelor Thesis - Daniel Speck

This repository contains the bachelor thesis of Daniel Speck (PDF)
as well as code examples that can be used for any academic purpose.
Please remember to cite the work.

The thesis proposes a new approach for localizing the ball in
RoboCup humanoid soccer. A deep neural architecture is used without
any preprocessing at all. The localization part gets solved by a
convolutional neural network that is trained with probability
distributions on full images. No sliding-window approach is used.
The thesis only proposes a concept for the tracking part, consisting of
a simple recurrent neural network that filters and stabilizes the
localization output and predicts future distributions.


### Balltracking for Robocup Soccer using Deep Neural Networks - Bachelor Thesis

The complete thesis can be viewed or downloaded here:
[thesis_ball_tracking_speck.pdf](https://gogs.mafiasi.de/12speck/BachelorThesis/src/master/thesis_ball_tracking_speck.pdf)




### Ball Localization for Robocup Soccer using Convolutional Neural Networks - Paper

The complete paper can be viewed or downloaded here:
[paper_ball_localization_speck.pdf](https://gogs.mafiasi.de/12speck/BachelorThesis/src/master/paper_ball_localization_speck.pdf)

##### Best Paper Award

The paper won the _Best Paper Award for Engineering Contribution_
at the 20th Annual RoboCup International Symposium 2016 in Leipzig, Germany.

See: http://www.robocup2016.org/en/symposium/best-paper-award

##### Cite - Bibtex

```
@inproceedings{speck2016robocup,
  title={Ball Localization for Robocup Soccer using
Convolutional Neural Networks},
  author={Speck, Daniel and Barros, Pablo and Weber, Cornelius and Wermter, Stefan},
  booktitle={RoboCup 2016: Robot World Cup XX},
  year={2016},
  series={Lecture Notes in Computer Science},
  organization={Springer International Publishing}
}
```

### Code - Setup

##### Python Modules

In order to run the code examples you will need a Python 2.7 interpreter
up and running as well as some additional Python libraries. Here's a list:

```
> pip freeze

cycler==0.10.0
enum34==1.1.6
funcsigs==1.0.2
h5py==2.6.0
matplotlib==1.5.3
mock==2.0.0
numpy==1.11.2
pbr==1.10.0
protobuf==3.0.0
pyparsing==2.1.10
python-dateutil==2.6.0
pytz==2016.7
six==1.10.0
tensorflow==0.11.0
```

##### Environment Variables

The code relies on the existance of three
[environment variables](https://en.wikipedia.org/wiki/Environment_variable),
which set up the paths to the executables, the data, and the log files.

```
BTLOG     # results and training info gets stored here
BTDATA    # training/test data directory
BTEX      # path to the repository / executables
```

You can set up environment variables directly via the terminal (non-permanent):

```
# example paths
export BTLOG="/home/your-user/speck-bachelor-thesis/log/"
export BTDATA="/home/your-user/speck-bachelor-thesis/data/"
export BTEX="/home/your-user/repositories/speck-bachelor-thesis/code/"
```

### Data

We [(Hamburg Bit-Bots)](https://robocup.informatik.uni-hamburg.de/en/) are currently working on a cloud solution for image labelling
that also allows downloading the data sets.

##### Trained Networks

I will release trained neural networks shortly. This will enable anyone to
load the pre-trained network and test it on your own data.
