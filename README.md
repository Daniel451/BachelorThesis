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

In order to run the code examples you will need a Python 2.7 interpreter
up and running as well as some additional Python libraries. Here's a list:

```
> pip freeze

appdirs==1.4.0
argcomplete==1.4.1
backports.shutil-get-terminal-size==1.0.0
beautifulsoup4==4.5.0
chardet==2.3.0
cycler==0.10.0
Cython==0.25.1
decorator==4.0.10
EbookLib==0.15
enum34==1.1.6
funcsigs==1.0.2
h5py==2.6.0
ipython==5.0.0
ipython-genutils==0.1.0
louis==3.0.0
lxml==3.6.1
matplotlib==1.5.1
mock==2.0.0
netsnmp-python==1.0a1
numpy==1.11.2
packaging==16.8
pathlib2==2.1.0
pbr==1.10.0
pdfminer==20140328
pexpect==4.2.0
pickleshare==0.7.3
Pillow==3.3.0
prompt-toolkit==1.0.3
protobuf==3.1.0
ptyprocess==0.5.1
pwquality==1.3.0
Pygments==2.1.3
pyparsing==2.1.5
python-dateutil==2.5.3
python-docx==0.8.6
python-libtorrent==1.1.1
python-musicbrainz==0.0.0
python-pptx==0.5.8
pytz==2016.6.1
simplegeneric==0.8.1
six==1.10.0
SpeechRecognition==3.4.6
team==1.0
tensorflow==0.11.0rc1
tesserocr==2.1.2
textract==1.4.0
traitlets==4.2.2
virtualenv==15.0.3
wcwidth==0.1.7
xlrd==1.0.0
XlsxWriter==0.9.3
```